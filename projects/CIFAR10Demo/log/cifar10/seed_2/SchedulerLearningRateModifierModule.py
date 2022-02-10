from ptschedulers.SchedulerBaseModule import SchedulerBaseModule

class SchedulerLearningRateModifierModule(SchedulerBaseModule):
    
    def __init__(self,
        schedule={},
        optimizer='optimizer_main',
        modules=[]):
        super().__init__()

        self.schedule = schedule
        self.optimizer = optimizer
        self.modules = modules


    def step(self, config, shared_modules, scheduler_modules_ordered_dict, epoch):
        if epoch not in self.schedule:
            return None

        optimizer = shared_modules[self.optimizer]
        new_lr = self.schedule[epoch]
        
        for i, param_group in enumerate(optimizer.param_groups):
            if not self.modules or i in self.modules:
                param_group['lr'] = new_lr
        print('changed learning rate to: {:g}'.format(new_lr))