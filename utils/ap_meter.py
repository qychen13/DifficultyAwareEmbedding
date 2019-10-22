import numpy
import copy

import torch.nn.functional as F


class APMeter(object):
    def __init__(self, num_classes=1000):
        self.num_classes = num_classes
        self.counts = numpy.zeros(num_classes)
        self.right_counts = numpy.zeros(num_classes)

    def add(self, opt, target):
        # move to cpu
        target = target.cpu().numpy()
        pre = opt.cpu().max(1)[1].squeeze().numpy()

        self.counts += numpy.histogram(target, self.num_classes,
                                       range=[0, self.num_classes])[0]
        right_pre = target[pre == target]
        self.right_counts += numpy.histogram(right_pre,
                                             self.num_classes, range=[0, self.num_classes])[0]

    def value(self):
        return self.right_counts / self.counts

    def reset(self):
        self.counts = numpy.zeros(self.num_classes)
        self.right_counts = numpy.zeros(self.num_classes)


class ClassErrorEvaluationMeter(object):
    def __init__(self, num_classes=1000):
        self.num_classes = num_classes
        """
        self.counts = numpy.zeros(num_classes)
        """
        self.positive_error_sum = numpy.zeros(num_classes)
        self.negative_error_sum = numpy.zeros(num_classes)
        self.counts = numpy.zeros(num_classes)

    def add(self, opt, target):
        probabilities = F.softmax(opt)
        probabilities = probabilities.data.cpu().numpy()
        target = target.cpu().numpy()

        # calculate error
        row_index = list(range(probabilities.shape[0]))
        error = copy.deepcopy(probabilities)
        error[row_index, target] = 0
        self.negative_error_sum += error.sum(0)
        probabilities[row_index, target] = 1 - probabilities[row_index, target]
        self.positive_error_sum += probabilities.sum(0) - error.sum(0)
        self.counts += numpy.histogram(target, self.num_classes,
                                       range=[0, self.num_classes])[0]

    def value(self):
        return 0.5 * (self.positive_error_sum / self.counts + self.negative_error_sum / (self.counts.sum()-self.counts))

    def reset(self):
        self.counts = 0
        self.positive_error_sum = numpy.zeros(self.num_classes)
        self.negative_error_sum = numpy.zeros(self.num_classes)
