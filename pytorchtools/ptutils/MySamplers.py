from torch.utils.data import Sampler
import numpy as np

class SubsetRandomSegmentSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement. About segment_size many samples of a certain class are sampled sequentially. This means that for a high segment_size only few classes are present in a batch.

    """

    def __init__(self, indices, label_to_indices, segment_size):
        self.indices = indices
        self.label_to_indices = label_to_indices
        self.segment_size = segment_size
        self.n_segments = int(np.floor(len(self.indices) / self.segment_size))

    def __iter__(self):
        class_sorted_sequence = []
        for label_inds in self.label_to_indices.values():
            inds = np.intersect1d(self.indices, label_inds)
            inds = inds[np.random.permutation(len(inds))]
            class_sorted_sequence.extend(inds)
        class_sorted_sequence = np.array(class_sorted_sequence)

        sort_segment_inds = np.random.permutation(self.n_segments)
        sort_inds = np.arange(self.n_segments * self.segment_size)
        sort_inds = sort_inds.reshape((self.n_segments, self.segment_size))
        sort_inds = sort_inds[sort_segment_inds]
        indices_sorted = class_sorted_sequence[sort_inds.ravel()]

        return (i for i in indices_sorted)

    def __len__(self):
        return  self.n_segments * self.segment_size

class SubsetSequentialSampler(Sampler):
    r"""Samples elements sequentially

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class ClassBalancedSampler(Sampler):
    
    def __init__(self, label_to_indices_orig, b_undersample=True, indices=None, n_samples_per_class=-1):
        #remove indices
        if indices is not None:
            self.label_to_indices = {key : np.array([i for i in item if i in indices], dtype=int) for key, item in label_to_indices_orig.items()} 
        else:
            self.label_to_indices = label_to_indices_orig
        self.n_classes = len(self.label_to_indices)
        self.class_counts = [len(self.label_to_indices[key]) for key in range(self.n_classes)]
        self.b_undersample = b_undersample
        
        if n_samples_per_class > 0:
            self.n_samples_per_class = n_samples_per_class
        elif b_undersample:
            #undersample
            self.n_samples_per_class = np.min(self.class_counts)
        else:
            #oversample
            self.n_samples_per_class = np.max(self.class_counts)

    def __iter__(self):
        inds_out = []
        for inds in self.label_to_indices.values():
            permutation = np.random.permutation(len(inds))
            if len(permutation) < self.n_samples_per_class:
                permutation = np.repeat(permutation, self.n_samples_per_class//len(permutation)+1)
            inds_out.extend(inds[permutation[:self.n_samples_per_class]])
        np.random.shuffle(inds_out)
        return iter(inds_out)

    def __len__(self):
        return self.n_samples_per_class * self.n_classes

