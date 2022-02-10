import torch.utils.data as data
import numpy as np

from ptutils.my_worker_init_fn import my_worker_init_fn
from ptutils.MySamplers import (SubsetRandomSegmentSampler, SubsetSequentialSampler, ClassBalancedSampler)


class DataloaderGetter(object):
    
    def __init__(self,
        dataset,
        batch_size=32,
        b_train_fold=None,
        fold=None, 
        n_folds=None, 
        b_stratified=True, 
        b_shuffle_before_splitting=True, 
        p_train=None, 
        sampler_segments=None, 
        drop_last=False, 
        b_shuffled=True, 
        use_only=-1, 
        n_train=None,
        custom_seed=42,
        random_sampler_num_samples=None, 
        b_undersample=False):
        super().__init__()

        self.dataset_name = dataset
        self.batch_size=batch_size
        self.b_train_fold=b_train_fold
        self.fold=fold
        self.n_folds=n_folds
        self.b_stratified=b_stratified
        self.b_shuffle_before_splitting=b_shuffle_before_splitting
        self.p_train=p_train
        self.sampler_segments=sampler_segments
        self.drop_last=drop_last
        self.b_shuffled=b_shuffled
        self.use_only=use_only
        self.n_train=n_train
        self.custom_seed=custom_seed
        self.random_sampler_num_samples=random_sampler_num_samples
        self.b_undersample=b_undersample


    def get_dataloader(self, cuda_args, shared_modules):
        
        dataset = shared_modules[self.dataset_name]
        
        if ((self.fold is not None) + (self.p_train is not None) + (self.n_train is not None)) > 1:
            raise ValueError('Wrong settings in fold, p_train or n_train.')

        if self.fold is not None:
            if self.b_stratified:
                indices = dataset.get_stratified_k_fold_indices(fold=self.fold, n_folds=self.n_folds, b_train=self.b_train_fold, seed=self.custom_seed)
            else:
                indices = dataset.get_k_fold_indices(fold=self.fold, n_folds=self.n_folds, b_train=self.b_train_fold, b_shuffle_before_splitting=self.b_shuffle_before_splitting, seed=self.custom_seed)
        elif self.p_train is not None or self.n_train is not None:
            if self.b_stratified:
                if self.p_train is not None: 
                    n_or_p_train = self.p_train
                else:
                    n_or_p_train = self.n_train
                indices = dataset.get_balanced_shuffle_split_indices(n_or_p_train=n_or_p_train, b_train=self.b_train_fold, seed=self.custom_seed)
            else:
                raise NotImplementedError()
        else:
            indices = np.arange(len(dataset))
        
        if self.use_only > 0:
            my_random_state = np.random.RandomState(self.custom_seed) #I made the seed none on 2021_01_19 and I set it to the seed in config on 2021_02_04 and made it to custom_seed on 21_05_
            use_only = min(self.use_only, len(indices))
            indices = indices[my_random_state.permutation(len(indices))[:use_only]]
        
        if self.sampler_segments is not None:
            sampler = SubsetRandomSegmentSampler(indices=indices, label_to_indices=dataset.label_to_indices, segment_size=self.sampler_segments)
        elif self.b_undersample:
            if not self.b_shuffled:
                raise ValueError("You cannot use undersampling and not shuffling.")
            sampler = ClassBalancedSampler(dataset.label_to_indices, b_undersample=True, indices=indices)
        elif self.random_sampler_num_samples is not None:
            #does not care about indices selected above
            sampler = data.sampler.RandomSampler(dataset, replacement=True, num_samples=self.random_sampler_num_samples)
        elif self.b_shuffled:
            sampler = data.sampler.SubsetRandomSampler(indices=indices)
        else:
            sampler = SubsetSequentialSampler(indices=indices)

        return data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            drop_last=self.drop_last,
            **cuda_args,
            worker_init_fn=my_worker_init_fn)