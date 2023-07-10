import torch.utils.data as data
import os
from shutil import copytree
import numpy as np
from PIL import Image
from multiprocessing import Manager
from typing import Tuple

class DatasetClassesFromFoldersNtrainNtest(data.Dataset):
    def __init__(self,
        datapath: str,
        transform: str = '',
        copy_data_to: str = '/data/tmp_data',
        convert_to_rbg_image: bool = True,
        tags: Tuple = {},
        n_train_per_class: int = -1,
        n_test_per_class: int = -1,
        train_fold: bool = True,
        custom_seed: int = 42):
        super().__init__()

        assert n_train_per_class > 0 or n_test_per_class > 0

        self.datapath = datapath
        self.transform_name = transform
        self.copy_data_to = copy_data_to
        self.convert_to_rbg_image = convert_to_rbg_image
        self.tags = tags
        self.n_train_per_class = n_train_per_class
        self.n_test_per_class = n_test_per_class
        self.train_fold = train_fold
        self.custom_seed = custom_seed
        
    def prepare(self, shared_modules):
        #load data to /tmp if not already there
        if self.copy_data_to:
            datapath_local = os.path.join(self.copy_data_to, self.datapath.split('/')[-1])
            if not os.path.isdir(datapath_local):
                copytree(self.datapath, datapath_local)
            self.datapath = datapath_local

        self.transform = None
        if self.transform_name:
            self.transform = shared_modules[self.transform_name]
        
        
        classfolder_paths = [os.path.join(self.datapath, name) for name in sorted(os.listdir(self.datapath)) if os.path.isdir(os.path.join(self.datapath, name))]
        self.n_classes = len(classfolder_paths)

        EXTENSIONS = set(['jpg', 'png', 'JPEG'])

        self.image_paths = []
        self.labels = []
        self.class_names = []
        randomstate = np.random.RandomState(self.custom_seed)

        for class_id, classfolder_path in enumerate(classfolder_paths):
            classname = classfolder_path.split('/')[-1]
            self.class_names.append(classname)
            image_paths_tmp = np.array([os.path.join(self.datapath, classname, name) for name in sorted(os.listdir(classfolder_path)) if name.split('.')[-1] in EXTENSIONS])
            n_samples_of_this_class = len(image_paths_tmp)
            permutation = randomstate.permutation(n_samples_of_this_class)
            n_train = self.n_train_per_class if self.n_train_per_class > 0 else n_samples_of_this_class - self.n_test_per_class
            n_test = self.n_test_per_class if self.n_test_per_class > 0 else n_samples_of_this_class - self.n_train_per_class
            if self.train_fold:
                self.image_paths.extend(image_paths_tmp[permutation[:n_train]])
                self.labels.extend([class_id for i in range(n_train)])
            else:
                self.image_paths.extend(image_paths_tmp[permutation[n_train:n_train + n_test]])
                self.labels.extend([class_id for i in range(n_test)])
            

        manager = Manager() # use manager to improve the shared memory between workers which load data. Avoids the effect of ever increasing memory usage. See https://github.com/pytorch/pytorch/issues/13246#issuecomment-445446603
        self.tags = manager.dict(self.tags)
        self.image_paths = manager.list(self.image_paths)
        self.labels = manager.list(self.labels)

        self.label_to_indices = {}
        labels_tmp = np.array(self.labels)
        for c in range(self.n_classes):
            self.label_to_indices[c] = np.where(labels_tmp==c)[0]
        

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        label = self.labels[idx]
        image_path = self.image_paths[idx]
        out_image = Image.open(image_path)
        if self.convert_to_rbg_image:
            out_image = out_image.convert('RGB')
        if self.transform is not None:
            out_image = self.transform(out_image)
        sample = {'data' : out_image, 'label' : label, 'id' : idx, 'tags' : dict(self.tags), 'path' : image_path}
        return sample