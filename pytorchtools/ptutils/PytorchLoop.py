import os
import sys
from tqdm import tqdm
import code
import torch
import ptutils.PytorchHelpers as ph
import select


class PytorchLoop(object):
    def __init__(self, config):

        self.config = config
        self.shared_modules = {}
        self.scheduler_modules_ordered_dict = {}

        # set up device 
        use_cuda = not self.config['no_cuda'] and torch.cuda.is_available()
        self.config['cuda_args'] = {'num_workers': self.config['num_workers'], 'pin_memory': self.config['pin_memory']} if use_cuda else {}
        self.shared_modules['device'] = torch.device(self.config['cuda_device']) if use_cuda else torch.device("cpu")

        # set up networks:
        if 'networks' in self.config:
            networks = ph.get_generic_modules(self.config, 'networks')
            for key in list(networks.keys()):
                networks[key] = networks[key].to(self.shared_modules['device'])
            self.shared_modules.update(networks)

        # set up losses
        if 'loss_fns' in self.config:
            loss_fns = ph.get_generic_modules(self.config, 'loss_fns')
            self.shared_modules.update(loss_fns)

        # set up optimizers, they will be properly created by the scheduler modules using the optimizers
        if 'optimizers' in self.config:
            optimizers = dict()
            for name in self.config['optimizers'].keys():
                optimizers[name] = None
            self.shared_modules.update(optimizers)

        # set up transforms
        if 'transforms' in self.config:
            transforms = ph.get_generic_modules(self.config, 'transforms')
            self.shared_modules.update(transforms)

        # set up datasets
        if 'datasets' in self.config:
            datasets = ph.get_generic_modules(self.config, 'datasets')
            for dataset in datasets.values():
                dataset.prepare(self.shared_modules)
            self.shared_modules.update(datasets)

        # set up loaders
        if 'loaders' in self.config:
            dataloaders = ph.get_generic_modules(self.config, 'loaders')
            dataloaders = {key : value.get_dataloader(self.config['cuda_args'], self.shared_modules) for key, value in dataloaders.items()}
            if self.config['save_data_paths']:
                for loadername, loader in dataloaders.items():
                    paths = ph.get_paths_from_loader(loader)
                    ph.write_array_to_file(paths, os.path.join(self.config['log_path'], loadername + '_paths.txt'))
            self.shared_modules.update(dataloaders)

        # set up scheduler
        if 'scheduler_modules' in self.config:
            self.scheduler_modules_ordered_dict = ph.get_generic_modules(self.config, 'scheduler_modules')

    def start_loop(self):
        #start loop
        if 'epochs' in self.config and self.config['epochs'] > 0:
            print('start main loop, start interactive mode by pressing i and Enter')
            for epoch in tqdm(range(1, self.config['epochs'] + 1)):
                try:
                    b_input = select.select([sys.stdin], [], [], 0)[0]
                    if b_input:
                        value = sys.stdin.readline().rstrip()
                        if (value == "i"):
                            print("Open Interactive Console, continue running with Strg+d")
                            code.interact(local=locals())
                        else:
                            print('received: ' + value)
                except:
                    print('failed to read input')
                for module in self.scheduler_modules_ordered_dict.values():
                    module.step(self.config, self.shared_modules, self.scheduler_modules_ordered_dict, epoch)

        #finish run
        for module in self.scheduler_modules_ordered_dict.values():
            module.finalize(self.config, self.shared_modules, self.scheduler_modules_ordered_dict)
