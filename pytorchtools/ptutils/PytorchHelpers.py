import os
import sys
import importlib.util as util 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import OrderedDict
        
def get_module(module_path):
    module_name = module_path.split("/")[-1][:-3]
    
    if module_path[0]!="/":
        module_path = '/'.join([os.getcwd(), module_path])
    
    spec = util.spec_from_file_location(module_name, module_path)
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def sequence_to_dict(items):
    """Transform a list containing a string key value sequence to a dictionary and cast to int or float if possible.   
    
    Arguments:
        items {list} -- list of strings like ['key1', 'value1', 'key2', 'value2', ...]
    
    Returns:
        dict -- Resulting dictionary with int or float values if possible
    """

    keys = items[0::2]
    values = items[1::2]
    for i, v in enumerate(values):
        values[i] = try_to_cast(v)
    return dict(zip(keys, values))

def is_float(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def try_to_cast(x):
    if x == 'True':
        return True
    elif x == 'False':
        return False
    elif is_float(x):
        if '.' in x:
            return float(x)
        elif 'nan' == x:
            return np.nan
        else:
            return int(x)
    else: 
        return x

def get_generic_modules(config, dict_attr, default_package='nn'):
    out_dict = OrderedDict()
    for module_name, config_of_module in config[dict_attr].items():
        if '.py' in config_of_module['source']:
            module = get_module(config_of_module['source'])
            inferred_class_name = config_of_module['source'].split("/")[-1][:-3]
            module_instance = getattr(module, inferred_class_name)(**config_of_module['params'])
        elif default_package == 'nn':
            module_instance = getattr(nn, config_of_module['source'])(**config_of_module['params'])
        out_dict[module_name] = module_instance
    return out_dict


def get_optimizer_for_network(config_of_module, param_groups):
    if '.py' in config_of_module['source']:
        module = get_module(config_of_module['source'])
        inferred_class_name = config_of_module['source'].split("/")[-1][:-3]
        optimizer = getattr(module, inferred_class_name)(param_groups, **config_of_module['params'])
    else:
        optimizer = getattr(optim, config_of_module['source'])(param_groups, **config_of_module['params'])
    return optimizer

def get_optimizer_for_tensors(config_of_module, tensors):
    if '.py' in config_of_module['source']:
        module = get_module(config_of_module['source'])
        inferred_class_name = config_of_module['source'].split("/")[-1][:-3]
        module_instance = getattr(module, inferred_class_name)(tensors, **config_of_module['params'])
    else:
        module_instance = getattr(optim, config_of_module['source'])(tensors, **config_of_module['params'])
    return module_instance

def concat(input_list):
    if isinstance(input_list[0], (torch.FloatTensor, torch.Tensor, torch.LongTensor)):
        #concatenate tensors
        return torch.cat(input_list,0)
    elif isinstance(input_list[0], list):
        if isinstance(input_list[0][0], (torch.FloatTensor, torch.Tensor, torch.LongTensor)):
            #there is a list with tensors inside. Concatenate each of these tensors, but keep list structure
            output_tensor_list = []
            for tensor_group in range(len(input_list[0])):
                output_tensor_list.append(torch.cat([batch_list[tensor_group] for batch_list in input_list], 0))
        else:
            #assume that we have a numpy array inside the list. Concatenate the numpy array and omit the outer list
            output_tensor_list = np.concatenate(input_list, 0)
        return output_tensor_list
    elif isinstance(input_list[0], dict):
        #there is a dict. We assume that there are tensors inside.
        output_tensor_dict = {}
        for key in input_list[0]:
            output_tensor_dict[key] = torch.cat([batch_dict[key] for batch_dict in input_list], 0)
        return output_tensor_dict
    else:
        raise ValueError('Output of model is neither tensor, list or dict')
    
def convert_to_device(value, device):
    try:
        if isinstance(value, (torch.FloatTensor, torch.Tensor, torch.LongTensor)):
            return value.to(device)
        elif isinstance(value, list):
            return [v.to(device) for v in value]
        elif isinstance(value, tuple):
            return tuple(v.to(device) for v in value)
        elif isinstance(value, dict):
            return {key : v.to(device) for key, v in value.items()}
    except:
        return value

def detach(value):
    try:
        if isinstance(value, (list, tuple)):
            return [x.detach() for x in value]
        elif isinstance(value, dict):
            return {key : x.detach() for key, x in value.items()}
        return value.detach()
    except:
        return value


"""
Write image paths to file 
"""

def get_paths_from_loader(dataloader):
    img_paths = []
    for batch in dataloader:
        if 'path' in batch:
            data = batch['path']
        else:
            data = batch['id']
        img_paths.extend(data)
    img_paths = np.array(img_paths)
    return img_paths


def write_array_to_file(iterable, path):
    with open(path, 'w') as f:
        for item in iterable:
            f.write(str(item)+"\n")