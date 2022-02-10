import os
import torch
import torch.nn as nn
import ptutils.PytorchHelpers as ph
from ptschedulers.SchedulerBaseModule import SchedulerBaseModule

class SchedulerTrainingInitializationModule(SchedulerBaseModule):
    """
    Create optimizer and store it in shared_modules under the fixed name 'optimizer_default', no matter the actual name of the optimizer in config.
    """
    def __init__(self,
        active_epochs=set([1]),
        network='network_main', 
        loss_fn='loss_fn_main', 
        loader='loader_train', 
        use_weights=False,
        optimizer='optimizer_main',
        param_groups_ids=[],
        set_all_weights_require_grad=False,
        set_bnorms_require_grad=False):
        super().__init__()

        self.active_epochs = active_epochs
        self.network = network
        self.loss_fn = loss_fn
        self.loader = loader
        self.use_weights = use_weights
        self.optimizer = optimizer
        self.param_groups_ids = param_groups_ids
        self.set_all_weights_require_grad = set_all_weights_require_grad
        self.set_bnorms_require_grad = set_bnorms_require_grad


        if not isinstance(self.active_epochs, set):
            self.active_epochs = set(self.active_epochs)

        
    def step(self, config, shared_modules, scheduler_modules_ordered_dict, epoch):
        if epoch not in self.active_epochs and epoch != -1:
            return None

        network = shared_modules[self.network]
        device = shared_modules['device']

        if self.set_bnorms_require_grad:
            for m in network.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    for param in m.parameters():
                        param.requires_grad = True

        if self.set_all_weights_require_grad:
            for param in network.parameters():
                param.requires_grad = True

        self.param_groups_ids.append('') #this is to include all parameters that are left into the last parameter group
        param_groups = []
        param_names_left = set([name for name, _ in network.named_parameters()])
        for identifier in self.param_groups_ids:
            params = []
            #in the case of multiple networks, we have to add the parameters of every network. 
            if isinstance(network, (list, tuple)):
                parameters_orig = [net.named_parameters() for net in network]
            else:
                parameters_orig = [network.named_parameters()]
            for item in parameters_orig:
                for name, param in item:
                    if param.requires_grad and name in param_names_left and identifier in name:
                        params.append(param)
                        param_names_left.remove(name)
            param_groups.append({'params' : params})

        shared_modules[self.optimizer] = ph.get_optimizer_for_network(config['optimizers'][self.optimizer], param_groups)
        
        if self.use_weights:
            weights = shared_modules[self.get_param('loader')].dataset.get_class_weights()
            weights = torch.from_numpy(weights).float()
            weights = weights.to(device)
            shared_modules[self.loss_fn].set_weights(weights)

        print('initialized training with optimizer:')
        print(shared_modules[self.optimizer])