from arguments.arguments_trainval import ArgumentsTrainVal
from models.construct_model import construct_model
from utils.construct_engine import construct_engine
from datasets.construct_dataset import construct_train_dataloaders
from utils.network import init_network

import torch.nn.functional as functional
from torch.optim.sgd import SGD
import torch

import copy
import math


def main():
    args = ArgumentsTrainVal().parse_args()

    print('***************************Arguments****************************')
    print(args)

    model, distribution = construct_model(args)
    print('--------------------------Model Info----------------------------')
    print(model)

    if args.resume_model is None:
        init_network(model)

    criterion = functional.cross_entropy
    optimizer = SGD(model.parameters(), args.learning_rate,
                    momentum=args.momentum, weight_decay=args.weight_decay)

    train_iterator, validate_iterator = construct_train_dataloaders(args)

    engine_args = [args.gpu_ids, model, criterion, distribution,
                   train_iterator, validate_iterator, optimizer]
    if args.num_classes == 1000 or args.num_classes == 1001:
        topk = [1, 5]
    else:
        topk = [1]

    # learning rate points
    lr_points = []
    if args.num_classes == 100:
        lr_points = [150, 225]
    elif args.num_classes == 1000 or args.num_classes == 1001:
        lr_points = [30, 60]
    print('==> Set lr_points for resnet54: {}'.format(lr_points))

    engine = construct_engine(*engine_args, checkpoint_iter_freq=args.checkpoint_iter_freq,
                              checkpoint_epoch_freq=args.checkpoint_epoch_freq,
                              checkpoint_save_path=args.checkpoint_save_path, iter_log_freq=args.iter_log_freq,
                              topk=topk, num_classes=args.num_classes, lambda_error=args.lambda_error,
                              environment=args.environment, lr_points=lr_points)

    if args.ada_train:
        engine.ada_train(args.maxepoch)
    else:
        engine.resume(args.maxepoch, args.resume_epoch, args.resume_iteration)


if __name__ == '__main__':
    main()
