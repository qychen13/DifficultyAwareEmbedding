'''
Motivated by torchnet
'''
import torchnet.engine as engine
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as functional

from tqdm import tqdm

import math
import sys


class Engine(engine.Engine):
    def __init__(self, gpu_ids, network, criterion, distribution, train_iterator, validate_iterator, optimizer,
                 cumulate_gradient=1):
        self.state = {
            'gpu_ids': gpu_ids,
            'network': network,
            'train_iterator': train_iterator,
            'validate_iterator': validate_iterator,
            'maxepoch': None,
            'criterion': criterion,
            'distribution': distribution,
            'optimizer': optimizer,
            'epoch': None,
            't': None,
            'train': True,
            'output': None,
            'loss': None,
            'cumulate_gradient': cumulate_gradient
        }

        # set cudnn bentchmark
        torch.backends.cudnn.benchmark = True

        super(Engine, self).__init__()

    def resume(self, maxepoch, epoch, t):
        state = self.state

        state['epoch'] = epoch
        state['t'] = t
        state['maxepoch'] = maxepoch
        state['train'] = True

        self.hook('on_start', state)

        while state['epoch'] < state['maxepoch']:
            self.hook('on_start_epoch', state)

            i = 0
            for sample in tqdm(state['train_iterator']):
                state['sample'] = sample
                self.hook('on_end_sample', state)
                ipt, target = state['sample'][0], state['sample'][1]

                def closure():
                    self.hook('on_start_forward', state)
                    if state['gpu_ids'] is not None:
                        output = nn.parallel.data_parallel(
                            state['network'], ipt, state['gpu_ids'])
                    else:
                        output = state['network'](ipt)
                    loss = state['criterion'](
                        output, target, weight=state['distribution']) / state['cumulate_gradient']
                    state['output'] = output
                    state['loss'] = loss
                    self.hook('on_end_forward', state)
                    loss.backward()

                    # to free memory in save_for_backward
                    state['output'] = None
                    state['loss'] = None
                    return loss

                if i % state['cumulate_gradient'] == 0:
                    state['optimizer'].zero_grad()

                closure()

                if (i + 1) % state['cumulate_gradient'] == 0:
                    state['optimizer'].step()

                self.hook('on_end_update', state)
                state['t'] += 1
                i += 1

            state['epoch'] += 1
            self.hook('on_end_epoch', state)
        self.hook('on_end', state)
        return state

    def train(self, maxepoch):
        self.resume(maxepoch, 0, 0)

    def validate(self):
        state = self.state

        state['train'] = False
        self.hook('on_start', state)

        with torch.no_grad():
            for sample in tqdm(state['validate_iterator']):
                state['sample'] = sample
                self.hook('on_end_sample', state)
                ipt, target = state['sample'][0], state['sample'][1]

                def closure():
                    if state['gpu_ids'] is not None:
                        opt = nn.parallel.data_parallel(
                            state['network'], ipt, state['gpu_ids'])
                    else:
                        opt = state['network'](ipt)

                    state['output'] = opt
                    loss = state['criterion'](
                        opt, target, weight=state['distribution'])
                    state['loss'] = loss
                    self.hook('on_end_forward', state)
                    # to free memory in save_for_backward
                    state['output'] = None
                    state['loss'] = None

                closure()

        self.hook('on_end_test', state)
        self.hook('on_end', state)
        return state

    def update_distribution(self):
        state = self.state
        state['train'] = False
        print('==>Calculating new distribution......')
        self.hook('on_start', state)

        with torch.no_grad():
            for sample in tqdm(state['validate_iterator']):
                state['sample'] = sample
                self.hook('on_end_sample', state)
                ipt, target = state['sample'][0], state['sample'][1]

                def closure():
                    if state['gpu_ids'] is not None:
                        opt = nn.parallel.data_parallel(
                            state['network'], ipt, state['gpu_ids'])
                    else:
                        opt = state['network'](ipt)
                    state['output'] = opt
                    loss = state['criterion'](
                        opt, target, weight=state['distribution'])
                    state['loss'] = loss
                    self.hook('on_end_forward', state)
                    # to free memory in save_for_backward
                    state['output'] = None
                    state['loss'] = None

                closure()

        self.hook('on_end', state)
        self.hook('on_update_distribution', state)

    def ada_train(self, maxepoch):
        self.update_distribution()
        self.train(maxepoch)
