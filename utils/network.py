import torch.nn as nn
from torchvision.models.alexnet import AlexNet
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import math


def init_network(model):
    print('==> Network initialization.')
    if isinstance(model, AlexNet) and hasattr(model, 'classifier100'):  # fine tune alex100 model
        print('==> Fine tune alexnet100 model')

        model_urls = {
            'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
        }
        load_partial_network(model, model_zoo.load_url(model_urls['alexnet']))
        # normal init classifier100
        model = model.classifier100
        print('==> Normal init classifier100.')

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.reset_parameters()
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


def load_partial_network(model, state_dict):
    """
    Lot of copy from load_state_dict
    """
    print('==> Load Partial Network...')
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            raise KeyError('unexpected key "{}" in state_dict'
                           .format(name))
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state[name].copy_(param)
        except:
            print('While copying the parameter named {}, whose dimensions in the model are'
                  ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                      name, own_state[name].size(), param.size()))
            raise

    missing = set(own_state.keys()) - set(state_dict.keys())
    print('******Not load {}******'.format(missing))
