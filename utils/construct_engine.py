'''
Motivated by tnt.example
'''
import os
import time
import numpy
import visdom
import copy

import torch
from torch.autograd.variable import Variable

from torchnet.logger import VisdomPlotLogger
import torchnet.meter as meter

from .engine import Engine
from .ap_meter import APMeter
from .network import init_network


def construct_engine(*engine_args, checkpoint_iter_freq=None, checkpoint_epoch_freq=1, checkpoint_save_path='checkpoints',
                     iter_log_freq=100, topk=[1, 5], num_classes=1000,
                     lambda_error=0.7, environment='main', lr_points=[], server='localhost'):
    engine = Engine(*engine_args)


    # meters
    time_meter = meter.TimeMeter(1)
    data_loading_meter = meter.MovingAverageValueMeter(windowsize=100)
    gpu_time_meter = meter.MovingAverageValueMeter(windowsize=100)

    classerr_meter = meter.ClassErrorMeter(topk)
    train_loss_meter = meter.MovingAverageValueMeter(windowsize=100)
    test_loss_meter = meter.AverageValueMeter()
    ap_meter = APMeter(num_classes)

    # logger associated with meters
    data_loading_logger = VisdomPlotLogger('line', server=server, opts={'title': 'Data Loading Time'}, env=environment)
    gpu_time_logger = VisdomPlotLogger('line', server=server, opts={'title': 'Gpu Computing Time'}, env=environment)
    classerr_meter_iter_loggers = []
    classerr_meter_epoch_logers = []
    for i in range(len(topk)):
        classerr_meter_iter_loggers.append(
            VisdomPlotLogger('line', server=server, opts={'title': 'Classification Top {} Error Along Iterations'.format(topk[i])},
                             env=environment))
        classerr_meter_epoch_logers.append(
            VisdomPlotLogger('line', server=server, opts={'title': 'Classification Top {} Error Along Epochs'.format(topk[i])},
                             env=environment))
    loss_meter_iter_logger = VisdomPlotLogger('line', server=server, opts={'title': 'Loss in One Iteration'}, env=environment)
    loss_meter_epoch_logger = VisdomPlotLogger('line', server=server, opts={'title': 'Loss with Epoch'}, env=environment)
    test_loss_logger = VisdomPlotLogger('line', server=server, opts={'title': 'test loss'}, env=environment)
    test_error_logger = VisdomPlotLogger('line', server=server, opts={'title': 'test error'}, env=environment)
    weighted_error_log = VisdomPlotLogger('line', server=server, opts={'title': 'weighted test error'}, env=environment)
    ap_logger = visdom.Visdom(env=environment, server='http://'+server)

    def prepare_network(state):
        # switch model
        if state['train']:
            state['network'].train()
        else:
            state['network'].eval()

    def wrap_data(state):
        if state['gpu_ids'] is not None:
            state['sample'][0] = state['sample'][0].cuda(device=state['gpu_ids'][0], async=False)
            state['sample'][1] = state['sample'][1].cuda(device=state['gpu_ids'][0], async=True)

        volatile = False

        if not state['train']:
            volatile = True

        if volatile:
            with torch.no_grad():
                state['sample'][0] = Variable(data=state['sample'][0])
                state['sample'][1] = Variable(data=state['sample'][1])
        else:
            state['sample'][0] = Variable(data=state['sample'][0])
            state['sample'][1] = Variable(data=state['sample'][1])


    def on_start(state):
        if state['gpu_ids'] is None:
            print('Training/Validating without gpus ...')
        else:
            if not torch.cuda.is_available():
                raise RuntimeError('Cuda is not available')

            state['network'].cuda(state['gpu_ids'][0])
            state['distribution'] = state['distribution'].cuda(state['gpu_ids'][0])
            print('Training/Validating on gpu: {}'.format(state['gpu_ids']))

        if state['train']:
            print('*********************Start Training at {}***********************'.format(time.strftime('%c')))
            if state['t'] == 0:
                filename = os.path.join(checkpoint_save_path, 'init_model.pth.tar')
                save_model(state, filename)
        else:
            print('-------------Start Validation at {} For Epoch{}--------------'.format(time.strftime('%c'),
                                                                                         state['epoch']))
        prepare_network(state)
        reset_meters()

    def on_start_epoch(state):
        reset_meters()
        print('--------------Start Training at {} for Epoch{}-----------------'.format(time.strftime('%c'),
                                                                                       state['epoch']))
        time_meter.reset()
        prepare_network(state)

    def on_end_sample(state):
        state['sample'].append(state['train'])
        wrap_data(state)
        data_loading_meter.add(time_meter.value())

    def on_start_forward(state):
        time_meter.reset()

    def on_end_forward(state):
        classerr_meter.add(state['output'].data, state['sample'][1].data)
        ap_meter.add(state['output'].data, state['sample'][1].data)
        if state['train']:
            train_loss_meter.add(state['loss'].data.item())
        else:
            test_loss_meter.add(state['loss'].data.item())
            

    def on_end_update(state):
        gpu_time_meter.add(time_meter.value())
        if state['t'] % iter_log_freq == 0 and state['t'] != 0:
            data_loading_logger.log(state['t'], data_loading_meter.value()[0])
            gpu_time_logger.log(state['t'], gpu_time_meter.value()[0])
            loss_meter_iter_logger.log(state['t'], train_loss_meter.value()[0])
            for i in range(len(topk)):
                classerr_meter_iter_loggers[i].log(state['t'], classerr_meter.value(topk[i]))
        if checkpoint_iter_freq and state['t'] % checkpoint_iter_freq == 0:
            filename = os.path.join(checkpoint_save_path,
                                    'e' + str(state['epoch']) + 't' + str(state['t']) + '.pth.tar')
            save_model(state, filename)
        time_meter.reset()

    def on_end_epoch(state):
        for i in range(len(topk)):
            classerr_meter_epoch_logers[i].log(state['epoch'], classerr_meter.value()[i])
        loss_meter_epoch_logger.log(state['epoch'], train_loss_meter.value()[0])
        print('***************Epoch {} done: class error {}, loss {}*****************'.format(state['epoch'],
                                                                                              classerr_meter.value(),
                                                                                              train_loss_meter.value()))
        if checkpoint_epoch_freq and state['epoch'] % checkpoint_epoch_freq == 0:
            filename = os.path.join(checkpoint_save_path,
                                    'e' + str(state['epoch']) + 't' + str(state['t']) + '.pth.tar')
            save_model(state, filename)
            # calculate sorted indexes w.r.t distribution
            sort_indexes = numpy.argsort(state['distribution'].cpu().numpy())
            ap_logger.line(X=numpy.linspace(0, num_classes, num=num_classes, endpoint=False),
                           Y=ap_meter.value()[sort_indexes], opts={'title': 'AP Change E{}(Training)'.format(state['epoch'])},
                           win='trainap{}'.format(state['epoch']))
        # adjust learning rate
        if state['epoch'] in lr_points:
            adjust_learning_rate(state)

        reset_meters()

        # do validation at the end of epoch
        state['train'] = False
        engine.validate()
        state['train'] = True

    def on_end_test(state):
        test_loss_logger.log(state['epoch'], test_loss_meter.value()[0])
        pre_distribution = state['distribution'].cpu().numpy()
        weighted_error = pre_distribution / pre_distribution.sum() * (1 - ap_meter.value())
        weighted_error = weighted_error.sum()
        weighted_error_log.log(state['epoch'], weighted_error)
        if checkpoint_epoch_freq and state['epoch'] % checkpoint_epoch_freq == 0:
            # calculate sort indexes w.r.t distribution
            sort_indexes = numpy.argsort(pre_distribution)
            ap_logger.line(X=numpy.linspace(0, num_classes, num=num_classes, endpoint=False),
                           Y=ap_meter.value()[sort_indexes], opts={'title': 'AP Change E{}(Test)'.format(state['epoch'])},
                           win='testap{}'.format(state['epoch']))
        for v in classerr_meter.value():
            test_error_logger.log(state['epoch'], v)
        print('----------------Test epoch {} done: class error {}, loss {}------------------'.format(state['epoch'],
                                                                                                     classerr_meter.value(),
                                                                                                     test_loss_meter.value()))
        reset_meters()

    def on_end(state):
        t = time.strftime('%c')
        if state['train']:
            print('*********************Training done at {}***********************'.format(t))
        else:
            print('*********************Validation done at {}***********************'.format(t))

    def on_update_distribution(state):

        # set info w.r.t the boost setting
        save_file_name = 'weak-learner.pth.tar'

        # calculate distribution w.r.t ap
        pre_distribution = state['distribution'].cpu().numpy()
        error = pre_distribution / pre_distribution.sum() * (1 - ap_meter.value())
        error = lambda_error * error.sum()
        beta = error / (1 - error)
        distribution = pre_distribution * numpy.power(beta, ap_meter.value())

        # normalization
        distribution = distribution / distribution.sum() * num_classes

        print('==> Calculating distribution done.')

        vis = visdom.Visdom(env=environment, server='http://'+server)
        vis.bar(X=distribution, opts={'title': 'Distribution'})

        # update model
        model = state['network']
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        weak_learner = {'beta': beta,
                        'model': model.state_dict(),
                        'distribution': state['distribution'],
                        'ap': ap_meter.value(),
                        'loss': test_loss_meter.value(),
                        'classerr': classerr_meter.value()}

        torch.save(weak_learner, os.path.join(checkpoint_save_path, save_file_name))
        print('==>Loss: {}'.format(weak_learner['loss']))
        print('==>Class Error: {}'.format(classerr_meter.value()))
        print('==>Beta: {}'.format(beta))
        print('==>{} saved.'.format(save_file_name))

        reset_meters()

        init_network(state['network'])

        # update distribution
        distribution = distribution.astype(numpy.float32)
        if state['gpu_ids'] is not None:
            distribution = torch.from_numpy(distribution).cuda(state['gpu_ids'][0])
        state['distribution'] = distribution
        if 'beta' in state.keys():
            state.pop('beta')


    def reset_meters():
        time_meter.reset()
        classerr_meter.reset()
        train_loss_meter.reset()
        test_loss_meter.reset()
        ap_meter.reset()

    def save_model(state, filename):
        model = state['network']
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        torch.save({'model': model.state_dict(), 'distribution': state['distribution']}, filename)
        print('==>Model {} saved.'.format(filename))

    def adjust_learning_rate(state):
        optimizer = state['optimizer']
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

        print('~~~~~~~~~~~~~~~~~~adjust learning rate~~~~~~~~~~~~~~~~~~~~')

    engine.hooks['on_start'] = on_start
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_sample'] = on_end_sample
    engine.hooks['on_start_forward'] = on_start_forward
    engine.hooks['on_end_forward'] = on_end_forward
    engine.hooks['on_end_update'] = on_end_update
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_end_test'] = on_end_test
    engine.hooks['on_end'] = on_end
    engine.hooks['on_update_distribution'] = on_update_distribution

    return engine
