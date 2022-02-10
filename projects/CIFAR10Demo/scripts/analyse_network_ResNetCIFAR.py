import matplotlib.pyplot as plt
from ptutils.PytorchLoop import PytorchLoop
import ptutils.analyse_network as an

SAVE_EXTENSION = "pdf"

# Use LaTeX text interpretation in figures
plt.rcParams.update({
    "text.usetex": True})

def get_config():

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
    item['source'] = '/nfshome/linse/Documents/development/pytorch_classification/pytorchtools/ptnetworks/ResNetCIFAR.py'
    item['params'] = {
        'variant' : 'resnet018',
        'n_classes' : 10, 
        'pretrained' : False,
        }

    return config

config = get_config()
pytorch_loop = PytorchLoop(config)
shared_modules = pytorch_loop.shared_modules


result = an.summarize_network(shared_modules['network_main'], (3, 32, 32))
print(result)