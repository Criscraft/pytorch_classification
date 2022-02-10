import os
import torch
import numpy as np
import json
from ptnetworks.ResNetCIFAR import ResNetCIFAR
from feature_visualization.NoiseGenerator import NoiseGenerator
from feature_visualization.FeatureVisualizerRobust import FeatureVisualizer


N_CLASSES = 10
IMAGE_SHAPE = (32, 32)
NORM_MEAN = [x / 255.0 for x in [125.3, 123.0, 113.9]]
NORM_STD = [x / 255.0 for x in [63.0, 62.1, 66.7]]
EXPORTPATH = '../results/feature_visualizations'
EXPORTINTERVAL = 50
EPOCHS = 200

log = {}
parameters = {}; log['parameters'] = parameters

lr_list = [0.1, 0.15, 0.2]; parameters['lr'] = lr_list
module_list = ["embedded_model.layer1", "embedded_model.layer2", "embedded_model.layer3", "embedded_model.layer4", "embedded_model.classifier"]; parameters['module_list'] = module_list
module_to_channel_dict = {}; parameters['module_to_channel_dict'] = module_to_channel_dict
n_channels = 4

use_cuda = torch.cuda.is_available()
device = torch.device("cuda") if use_cuda else torch.device("cpu")

model = ResNetCIFAR(
    variant='resnet018',
    n_classes=N_CLASSES, 
    pretrained=False,
    statedict='../log/cifar10/seed_0/model_00-1.pt')
model = model.to(device)
for param in model.embedded_model.parameters():
    param.requires_grad = False
model.eval()

"""
for name, module in model.named_modules():
    print(name)
    print(module)
"""

def get_submodule(model, target):
        atoms = target.split(".")
        mod = model
        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(mod._get_name() + " has no attribute `" + item + "`")
            mod = getattr(mod, item)
            if not isinstance(mod, torch.nn.Module):
                raise AttributeError("`" + item + "` is not an nn.Module")
        return mod

vis_list = []; log['visualizations'] = vis_list

print("start generating feature visualizations")
for module_name in module_list:
    channels = np.random.randint(0, 64, n_channels)
    module_to_channel_dict[module_name] = [int(item) for item in channels]
    for lr in lr_list:
        module = get_submodule(model, module_name)
        noise_generator = NoiseGenerator(device, IMAGE_SHAPE)
        init_image = noise_generator.get_noise_image()

        feature_visualizer = FeatureVisualizer(
            export=True,
            export_path=EXPORTPATH,
            export_interval=EXPORTINTERVAL,
            target_size=IMAGE_SHAPE,
            epochs=EPOCHS,
            lr=lr,
            )
        
        _, meta_info = feature_visualizer.visualize(model, module, device, init_image.detach().clone(), n_channels, channels)
        for item in meta_info:
            item.update({
                'module_name' : module_name,
                'lr': lr,
                })
            vis_list.append(item)

epochs = list(range(0, EPOCHS, EXPORTINTERVAL)) + [EPOCHS - 1]
log['parameters']['epochs'] = epochs
json_object = json.dumps(log, indent=4, ensure_ascii=False)
with open(os.path.join(EXPORTPATH, "meta.json"), "w") as outfile:
    outfile.write(json_object)
    
print("finished")
