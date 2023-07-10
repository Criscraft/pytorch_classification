import enum
import torch.nn as nn
import ptutils.PytorchHelpers as ph
from ptschedulers.SchedulerBaseModule import SchedulerBaseModule

class Mode(enum.Enum):
        CONV = "CONV"
        LINEAR = "LINEAR"
        BN = "BN"

def mode_to_module(mode):
    if mode == Mode.CONV:
        return nn.Conv2d
    elif mode == Mode.LINEAR:
        return nn.Linear
    elif mode == Mode.BN:
        return nn.BatchNorm2d

class SchedulerTrainingInitializationModule(SchedulerBaseModule):
    """
    Create optimizer and store it in shared_modules.
    """
    def __init__(self,
        active_epochs=set([1]),
        network='network_main',
        optimizer='optimizer_main',
        weight_decay=0.,
        modules_for_weight_decay=["CONV", "LINEAR", "BN"],
        weight_decay_applies_to_bias=True):
        super().__init__()

        self.active_epochs = active_epochs
        self.network = network
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        modules_for_weight_decay = [Mode(item) for item in modules_for_weight_decay]
        self.modules_for_weight_decay = tuple(mode_to_module(mode) for mode in modules_for_weight_decay)
        print(modules_for_weight_decay)
        print(self.modules_for_weight_decay)
        self.weight_decay_applies_to_bias = weight_decay_applies_to_bias

        if not isinstance(self.active_epochs, set):
            self.active_epochs = set(self.active_epochs)

        
    def step(self, config, shared_modules, scheduler_modules_ordered_dict, epoch):
        if epoch not in self.active_epochs and epoch != -1:
            return None

        network = shared_modules[self.network]

        optimizer_config = config['optimizers'][self.optimizer]
        if self.weight_decay > 0.:
            assert not hasattr(optimizer_config["params"], "weight_decay")
            # separate out all parameters to those that will and won't experience regularizing weight decay
            decay = set()
            all_p = set()
            no_decay = set()

            for mn, m in network.named_modules():
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                    all_p.add(fpn)
                     
                    if isinstance(m, self.modules_for_weight_decay):
                        if pn.endswith('bias') and not self.weight_decay_applies_to_bias:
                            continue
                        print("Add module: ")
                        print(fpn)
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
            
            no_decay = all_p - decay

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in network.named_parameters()}
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params), )

            # create the pytorch optimizer object
            param_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.weight_decay},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
        else:
            param_groups = [{"params": network.parameters()}]
            
        shared_modules[self.optimizer] = ph.get_optimizer_for_network(optimizer_config, param_groups)

        print('initialized training with optimizer:')
        print(shared_modules[self.optimizer])