import os
import torch
import numpy as np
import ptutils.PytorchHelpers as ph
from ptschedulers.SchedulerBaseModule import SchedulerBaseModule

class SchedulerTrainingModule(SchedulerBaseModule):

    def __init__(self,
        interval=1,
        network='network_main',
        loss_fn='loss_fn_main',
        optimizer='optimizer_main',
        loader='loader_train',
        min_virtual_batch_size=1,
        b_use_weights=False,
        id_target_layer='first_layer',
        filename='trainstatistics.dat'):
        super().__init__()

        self.interval = interval
        self.network = network
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.loader = loader
        self.min_virtual_batch_size = min_virtual_batch_size
        self.b_use_weights = b_use_weights
        self.id_target_layer = id_target_layer
        self.filename = filename

        self.log_keys = []

    
    def step(self, config, shared_modules, scheduler_modules_ordered_dict, epoch):
        if epoch % self.interval != 0:
            return None
        network = shared_modules[self.network]
        network.train()
        device = shared_modules['device']
        optimizer = shared_modules[self.optimizer]
        optimizer.zero_grad()
        loss_fn = shared_modules[self.loss_fn]
        loader = shared_modules[self.loader]
        
        n_minibatches = np.ceil(self.min_virtual_batch_size / loader.batch_size)
        mini_batch_counter = 0
        
        stats_minibatch = {}
        stats_batches = {}
        mean_gradient = None

        for batch in loader:
            mini_batch_counter += 1
            batch_size = batch['data'].shape[0]

            batch = {key : ph.convert_to_device(value, device) for key, value in batch.items()}
            loss_outputs = loss_fn(config, device, batch, network)
            
            if loss_outputs['loss'].item() > 1e-12:
                (loss_outputs['loss'] / loss_outputs['batch_size'] / n_minibatches).backward()
                
                if mini_batch_counter == n_minibatches:
                    optimizer.step()

                    #record the gradient of the target layer if possible
                    record_gradient = False
                    for m in network.modules():
                        if hasattr(m, 'ID') and self.id_target_layer in m.ID and m.weight.grad is not None:
                            gradient_of_target_layer = m.weight.grad.clone().detach()
                            record_gradient = True
                            break
                    if record_gradient:
                        parameter_size = torch.tensor([gradient_of_target_layer.numel()], dtype=torch.float, device=device, requires_grad=False) 
                        gradient_of_target_layer = gradient_of_target_layer.flatten()
                        gradient_of_target_layer_magnitude = gradient_of_target_layer.norm()
                        loss_outputs['grad_target_layer'] = gradient_of_target_layer_magnitude * batch_size / torch.sqrt(parameter_size)
                        if mean_gradient is None:
                            mean_gradient = torch.ones_like(gradient_of_target_layer, device=device)
                            mean_gradient = mean_gradient / (mean_gradient.norm() + 1e-6)
                        scalar_product = ((gradient_of_target_layer*mean_gradient).sum())/((gradient_of_target_layer.norm()+1e-6)*mean_gradient.norm())
                        alpha = torch.acos(scalar_product) / (2*3.141)*360
                        mean_gradient = exp_moving_average(mean_gradient, gradient_of_target_layer/(gradient_of_target_layer_magnitude+1e-6))
                        loss_outputs['grad_target_layer_alpha'] = alpha * batch_size

                    loss_outputs['lr'] = torch.tensor([optimizer.param_groups[0]['lr']], device=device) * batch_size

                    for key, item in loss_outputs.items():
                        if not key in stats_minibatch:
                                stats_minibatch[key] = 0.
                        stats_minibatch[key] += item.item()

                    for key, item in stats_minibatch.items():
                        if not key in stats_batches:
                            stats_batches[key] = []
                        stats_batches[key].append(item)

                    stats_minibatch = {}
                    mini_batch_counter = 0
                    optimizer.zero_grad()
                else:
                    for key, item in loss_outputs.items():
                        if not key in stats_minibatch:
                                stats_minibatch[key] = 0.
                        stats_minibatch[key] += item.item()
            else:
                print("loss equals zero, skip batch in epoch {:d}".format(epoch))
            
            del batch, loss_outputs

        if stats_batches:
            _filename = os.path.join(config['log_path'], self.filename)
            stats_batches = self.prepare_stats_batches(stats_batches, epoch)
            if not os.path.exists(_filename):
                self.log_keys = sorted(stats_batches.keys())
                self.init_log(_filename, self.log_keys)
            self.write_log(_filename, self.log_keys, stats_batches)
        else:
            print("skip epoch because the loss was always zero")
            
    def prepare_stats_batches(self, stats_batches, epoch):
            n_samples = np.array(stats_batches['batch_size']).sum()
            effective_batch_size = np.array(stats_batches['batch_size']).mean()
            stats_batches = {key: np.array(value).sum()/n_samples for key, value in stats_batches.items()} #here, batch_size key is mistreated
            stats_batches['batch_size'] = effective_batch_size #correct the mistreated batch_size
            stats_batches['epoch'] = epoch
            stats_batches['n_samples'] = n_samples
            return stats_batches

    def finalize(self, config, shared_modules, scheduler_modules_ordered_dict):
        if 'epochs' not in config:
            self.step(config, shared_modules, scheduler_modules_ordered_dict, 1)


    def init_log(self, filename, log_keys):
        textline = "\t".join(log_keys)
        with open(filename, 'w') as data:
            data.write("".join([textline, "\n"]))


    def write_log(self, filename, log_keys, stats_batches):
        textline = '\t'.join(['{:g}'.format(stats_batches[key]) for key in log_keys])
        with open(filename, 'a') as data:
            data.write("".join([textline, "\n"]))


def exp_moving_average(average, x, mu=0.8):
    if average is None:
        average = x
    return (1.0 - mu) * x + mu * average