import numpy as np
import random
import torch 

def my_worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % (2 ** 32))
    random.seed(torch.initial_seed() % (2 ** 32))