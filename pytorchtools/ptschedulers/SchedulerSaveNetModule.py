import os
from ptschedulers.SchedulerBaseModule import SchedulerBaseModule

class SchedulerSaveNetModule(SchedulerBaseModule):
    
    def __init__(self,
        active_epochs=set(),
        active_at_end=True,
        network='network_main',
        filename_base='model'):
        super().__init__()

        self.active_epochs = active_epochs
        self.active_at_end = active_at_end
        self.network = network
        self.filename_base = filename_base

        if not isinstance(self.active_epochs, set):
            self.active_epochs = set(self.active_epochs)
        
        
    def step(self, config, shared_modules, scheduler_modules_ordered_dict, epoch, force_step=False):
        if epoch not in self.active_epochs and not force_step:
            return None

        model = shared_modules[self.network]

        model.save(os.path.join(config['log_path'], "".join([self.filename_base, '_{:>04}'.format(epoch), '.pt'])))

        print('successfully saved model to disk')


    def finalize(self, config, shared_modules, scheduler_modules_ordered_dict):
        if self.active_at_end:
            if 'epochs' in config:
                epoch = config['epochs']
            else:
                epoch = -1
            self.step(config, shared_modules, scheduler_modules_ordered_dict, epoch, force_step=True)