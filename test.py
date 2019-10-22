from arguments.arguments_test import ArgumentsTest
from models.construct_model import construct_model
from utils.tester import test
from datasets.construct_dataset import construct_test_dataloaders
import torch

import math


def main():
    args = ArgumentsTest().parse_args()

    print('***************************Arguments****************************')
    print(args)
    args.resume_model = None

    weights = []
    models = []

    for i in range(len(args.test_models)):
        file = torch.load(args.test_models[i], lambda storage, loc: storage)

        beta = file['beta']
        weights.append(-math.log2(beta))

        model, _ = construct_model(args)
        model.load_state_dict(file['model'])
        print('==> Resume Model {}'.format(args.test_models[i]))

        models.append(model)

    print('--------------------------Model Info----------------------------')
    print(models[0])

    print('-------------------------- Weights -------------------------------')
    print(weights)

    topk = [1]
    if args.num_classes == 1000:
        topk = [1, 5]

    test_iterator = construct_test_dataloaders(args)

    test(models, weights, args.gpu_ids, iterator=test_iterator, topk=topk, num_classes=args.num_classes,
         enviroment=args.environment)


if __name__ == '__main__':
    main()
