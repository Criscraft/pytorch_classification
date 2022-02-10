import os
import torch
import torch.nn.functional as F
import pickle as pkl
import ptutils.PytorchHelpers as ph
from ptschedulers.SchedulerBaseModule import SchedulerBaseModule


class SchedulerLogOutputModule(SchedulerBaseModule):
    
    def __init__(self,
        active_epochs=set(), 
        active_at_end=True, 
        network='network_main', 
        loader='loader_test', 
        n_samples=1e8,
        save_thumbnails=False, 
        thumbnail_size=(32,32), 
        log_original_data=False, 
        keys_to_log_from_original_data_anyways=['label', 'id', 'path'], 
        filename='outputs.pkl'):
        super().__init__()

        self.active_epochs = active_epochs
        self.active_at_end = active_at_end
        self.network = network
        self.loader = loader
        self.n_samples = n_samples
        self.save_thumbnails = save_thumbnails
        self.thumbnail_size = thumbnail_size
        self.log_original_data = log_original_data
        self.keys_to_log_from_original_data_anyways = keys_to_log_from_original_data_anyways
        self.filename = filename
        
        if not isinstance(self.active_epochs, set):
            self.active_epochs = set(self.active_epochs)
        
    
    def step(self, config, shared_modules, scheduler_modules_ordered_dict, epoch, force_step=False):
        if epoch not in self.active_epochs and not force_step:
            return None
        
        if isinstance(self.network, (list, tuple)):
            models = [shared_modules[name] for name in self.network]
        else:
            models = [shared_modules[self.network]]
        for model in models:
            model.eval()
        loader = shared_modules[self.loader]
        device = shared_modules['device']
        output_dict = {}
        if self.save_thumbnails:
            output_dict['thumbnail'] = []

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i*loader.batch_size >= self.n_samples:
                    break
                batch = {key : ph.convert_to_device(value, device) for key, value in batch.items()}
                output_batch = batch
                for model in models:
                    output_batch = model(output_batch)

                if self.log_original_data:
                    for key, value in batch.items():
                        output_batch[key + '_orig'] = value
                else:
                    for key in self.keys_to_log_from_original_data_anyways:
                        output_batch[key + '_orig'] = batch[key]
                if self.save_thumbnails:
                    thumbnails = F.interpolate(batch['data'], size=self.thumbnail_size, mode='bilinear', align_corners=True)
                    output_batch['thumbnail'] = thumbnails

                for key, value in output_batch.items():
                    if key not in output_dict:
                        output_dict[key] = []
                    value = ph.convert_to_device(value, 'cpu')
                    output_dict[key].append(value)
                    
        output_dict = {key : ph.concat(value) for key, value in output_dict.items()}
        output_dict['epoch'] = epoch

        _filename = os.path.join(config['log_path'], self.filename)
        with open(_filename, 'ab') as datafile:
            pkl.dump(output_dict, datafile)

    def finalize(self, config, shared_modules, scheduler_modules_ordered_dict):
        if self.active_at_end:
            self.step(config, shared_modules, scheduler_modules_ordered_dict, -1, force_step=True)