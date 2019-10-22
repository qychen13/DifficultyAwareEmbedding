import argparse


class ArgumentsBase(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        parser = self.parser
        # gpu id
        parser.add_argument('-gs', '--gpu-ids', type=int, nargs='+',
                            help='multiple gpu device ids to train the network')

        # dataset setting
        parser.add_argument('-dir', '--directory', required=True,
                            help='dataset directory', metavar='DIR')
        parser.add_argument('-b', '--batch-size', required=True,
                            type=int, help='mini-batch size')
        parser.add_argument('-nw', '--num-workers', default=4,
                            type=int, help='workers for loading data')

        # visdom logfile setting
        parser.add_argument('-ilog', '--iter-log-freq', type=int,
                            default=100, help='log frequency over iterations')
        parser.add_argument('-en', '--environment', type=str,
                            default='main', help='log environment for visdom')

        # model setting
        parser.add_argument('-model', '--model-name', type=str,
                            choices=['mlp', 'resnet56', 'resnet50'], required=True,
                            help='model name')
        parser.add_argument('-nc', '--num_classes', type=int,
                            default=1000, help='number of the classes')

    def parse_args(self):
        return self.parser.parse_args()
