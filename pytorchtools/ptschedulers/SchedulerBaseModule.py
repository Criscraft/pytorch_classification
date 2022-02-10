class SchedulerBaseModule(object):
    def __init__(self):
        super().__init__()

    def step(self, args, shared_modules, scheduler_modules_ordered_dict, epoch):
        pass

    def finalize(self, args, shared_modules, scheduler_modules_ordered_dict):
        pass

