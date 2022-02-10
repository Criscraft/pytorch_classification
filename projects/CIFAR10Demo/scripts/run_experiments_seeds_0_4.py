import copy
import ptutils.PytorchHelpers as ph
from ptutils.start_task import start_task
import argparse

def get_tasks():
    
    task_names = []
    task_operations = []

    for seed in range(5):
        task_names.append('/seed_{}'.format(seed))
        task_operations.append([
            (['seed'], seed),
            ])
        
    return task_names, task_operations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str,
                        help="path to config file")
    parser.add_argument("-g", "--gpu", type=int, default=0,
                        help="index of gpu to use")
    params = parser.parse_args()

    config_path = params.config_path
    config_module = ph.get_module(config_path)

    task_names, task_operations = get_tasks()
    for (task_name, task_operation) in zip(task_names, task_operations):  
        config = config_module.get_config(gpu=params.gpu)
        for module_path, value in task_operation:
            set_config_item(config, module_path, value)
        config['log_path'] = config['log_path'] + task_name

        start_task(copy.deepcopy(config), copy.copy(config_path))


def set_config_item(config, module_path, value):
    dict_item = config
    for path in module_path[:-1]:
        dict_item = dict_item[path]
    dict_item[module_path[-1]] = value

if __name__ == '__main__':
    main()