import time
import numpy
import visdom
from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn
import torchnet.meter as meter
import torch.nn.functional as functional

from .ap_meter import APMeter


def test(models, weights, gpu_ids, iterator, topk, num_classes, enviroment='main'):
    print('=========================Start Testing at {}==========================='.format(
        time.strftime('%c')))

    # TODO: serialization
    classerr_meters = [meter.ClassErrorMeter(topk) for i in models]
    ap_meters = [APMeter(num_classes) for i in models]

    # multiple gpu support
    if gpu_ids is not None:
        for i in range(len(models)):
            models[i].cuda(gpu_ids[0])
            models[i] = torch.nn.DataParallel(models[i], device_ids=gpu_ids)

    # set eval() to freeze running mean and running var
    for m in models:
        m.eval()

    with torch.no_grad():
        for sample in tqdm(iterator):
            # wrap data
            for i in range(2):
                if gpu_ids is not None:
                    sample[i].cuda(gpu_ids[0], non_blocking=True)

            ipt, target = sample[0], sample[1]

            opt = None
            for i in range(len(models)):
                if opt is None:
                    opt = weights[i] * functional.softmax(models[i](ipt))

                else:
                    opt += weights[i] * functional.softmax(models[i](ipt))

                classerr_meters[i].add(opt.data, target.data)
                ap_meters[i].add(opt.data, target.data)

    # sorting w.r.t the first weak learner
    index = numpy.argsort(ap_meters[0].value())

    classerrs = []
    for i in topk:
        classerrs.append([meter.value(i) for meter in classerr_meters])
    ap = [meter.value()[index] for meter in ap_meters]
    ap = numpy.stack(ap)

    x = [numpy.linspace(0, num_classes, num=num_classes,
                        endpoint=False) for i in ap_meters]
    x = numpy.stack(x)

    vis = visdom.Visdom(server='http://localhost', env=enviroment)
    vis.line(X=x.transpose(), Y=ap.transpose(),
             opts={'title': 'Class AP'})
    for i in range(len(topk)):
        vis.line(numpy.asarray(classerrs[i]), opts={
                 'title': 'Class Top {} Error'.format(topk[i])})

    print('========================Testing Down at {} ==========================='.format(
        time.strftime('%c')))
    print('******************Top {} Error: {}*****************'.format(topk, classerrs))
