import os
import torch
import pandas as pd
from ptutils.PytorchLoop import PytorchLoop
from torchinfo import summary


def summarize_network(network, input_size):
    device = torch.device("cpu")
    network = network.to(device)
    pd.set_option("max_colwidth", 150)
    return summary(network, input_size, col_names=["kernel_size", "output_size", "num_params", "mult_adds"])


def print_modules(network):
    for name, item in network.named_modules():
        print(str(name) + '     ' + str(item))
        print('-----')

def get_config():

    pytorchtools_path = os.path.expanduser('~/Documents/development/pytorch_classification/pytorchtools')
    script_dir_path = os.path.dirname(os.path.realpath(__file__))

    config = {
        'seed' : 42,
        'num_workers' : 0,
        'pin_memory' : False,
        'no_cuda' : True,
        'cuda_device' : 'cpu',
        'save_data_paths' : False,
    }


    #networks
    config['networks'] = {}

    item = {}; config['networks']['network_main'] = item
    item['source'] = os.path.join(pytorchtools_path, 'ptnetworks/ResNetCIFAR.py')
    item['params'] = {
        'variant' : 'resnet018',
        'n_classes' : 10, 
        'pretrained' : False,
        }

    return config


config = get_config()
pytorch_loop = PytorchLoop(config)
shared_modules = pytorch_loop.shared_modules

result = summarize_network(shared_modules['network_main'], (1, 3, 224, 224))
print(result)