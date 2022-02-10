import os
import torch
import numpy as np
import ptutils.PytorchHelpers as ph
from ptschedulers.SchedulerBaseModule import SchedulerBaseModule

class SchedulerValidationModule(SchedulerBaseModule):

    def __init__(self,
        active_epochs=set(),
        active_at_end=True, 
        network='network_main',
        loss_fn='loss_fn_main',
        loader='loader_test',
        filename='validation.dat'):
        super().__init__()

        self.active_epochs = active_epochs
        self.active_at_end=active_at_end
        self.network = network
        self.loss_fn = loss_fn
        self.loader = loader
        self.filename = filename
        if not isinstance(self.active_epochs, set):
            self.active_epochs = set(self.active_epochs)

        self.log_keys = []
        
    
    def step(self, config, shared_modules, scheduler_modules_ordered_dict, epoch, force_step=False):
        if epoch not in self.active_epochs and not force_step:
            return None
        
        network = shared_modules[self.network]
        network.eval()
        device = shared_modules['device']
        loss_fn = shared_modules[self.loss_fn]
        loader = shared_modules[self.loader]
        
        stats_batches = {}

        with torch.no_grad():
            for batch in loader:
                batch = {key : ph.convert_to_device(value, device) for key, value in batch.items()}

                loss_outputs = loss_fn(config, device, batch, network)

                for key, item in loss_outputs.items():
                    if not key in stats_batches:
                        stats_batches[key] = []
                    stats_batches[key].append(item.item())

        stats_batches = self.prepare_stats_batches(stats_batches, epoch)
        _filename = os.path.join(config['log_path'], self.filename)
        if not os.path.exists(_filename):
            self.log_keys = sorted(stats_batches.keys())
            self.init_log(_filename, self.log_keys)
        self.write_log(_filename, self.log_keys, stats_batches)


    def prepare_stats_batches(self, stats_batches, epoch):
            n_samples = np.array(stats_batches['batch_size']).sum()
            effective_batch_size = np.array(stats_batches['batch_size']).mean()
            stats_batches = {key: np.array(value).sum()/n_samples for key, value in stats_batches.items()} #here, batch_size key is mistreated
            stats_batches['batch_size'] = effective_batch_size #correct the mistreated batch_size
            stats_batches['epoch'] = epoch
            stats_batches['n_samples'] = n_samples
            return stats_batches


    def finalize(self, config, shared_modules, scheduler_modules_ordered_dict):
        if self.active_at_end:
            if 'epochs' in config:
                epoch = config['epochs']
            else:
                epoch = -1
            self.step(config, shared_modules, scheduler_modules_ordered_dict, epoch, force_step=True)


    def init_log(self, filename, log_keys):
        textline = "\t".join(log_keys)
        with open(filename, 'w') as data:
            data.write("".join([textline, "\n"]))


    def write_log(self, filename, log_keys, stats_batches):
        textline = '\t'.join(['{:g}'.format(stats_batches[key]) for key in log_keys])
        with open(filename, 'a') as data:
            data.write("".join([textline, "\n"]))