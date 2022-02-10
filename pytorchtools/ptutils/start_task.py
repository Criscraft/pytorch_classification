import os
from shutil import copyfile
import random
import torch
import numpy as np
import json
from ptutils import PytorchLoop

def start_task(config, config_path):
    if os.path.exists(config['log_path']):
        config['log_path'] = config['log_path'] + '_new'
        return start_task(config, config_path)

    os.makedirs(config['log_path'])
    copyfile(config_path, os.path.join(config['log_path'], 'config_might_be_changed.py'))
    with open(os.path.join(config['log_path'], 'config.json'), "w") as f:
        f.write(json.dumps(config, indent=3, sort_keys=False))

    for value in config.values():
        if isinstance(value, dict):
            for sub_value in value.values():
                if 'source' in sub_value and '.py' in sub_value['source']:
                    copyfile(sub_value['source'], os.path.join(config['log_path'], sub_value['source'].split('/')[-1]))

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    print("Begin working on task. Write output to " + config['log_path'])
    loop = PytorchLoop.PytorchLoop(config)
    loop.start_loop()