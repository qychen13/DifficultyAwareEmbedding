import torch
import torchvision.models as tvmodels

from .cifar_resnet import ResNet
from .mlp import MLP

def construct_model(args):
    if args.model_name == 'resnet50':
        model = tvmodels.resnet.resnet50(False)
    elif args.model_name == 'resnet56':
        model = ResNet()
    elif args.model_name == 'MLP':
        model = MLP()
    else:
        raise NotImplementedError

    # default distribution, normalized version
    distribution = torch.Tensor(args.num_classes).fill_(1)
    if args.resume_model is not None:
        resume_model = torch.load(
            args.resume_model, map_location=lambda storage, loc: storage)

        # model containing distribution
        if 'distribution' in resume_model.keys():
            distribution = resume_model['distribution']
            resume_model = resume_model['model']
            print('==> Resume distribution')
        model.load_state_dict(resume_model)
        print('==> Resume from model {}.'.format(args.resume_model))
    else:
        print('==> Not init network!')
    return model, distribution
