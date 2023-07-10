# Codebase for image classification in Pytorch

This repository contains a CNN based classification pipeline for Pytorch. The directory 'pytorchtools' contains scripts for data analysis, neural network implementation, training loop and evaluation. The directory 'projects' contains specific experiments. In the demo project ResNet018 is trained and tested on CIFAR10.

## Setup

The required python packages can be found in the file requirements.txt
You can install them with 
    pip install -r requirements.txt 
Then, add the pytorchtools directory to your PYTHONPATH. This step is recommended if you want to use the code from anywhere in your system.

## Get started with the demo

The training configuration including hyperparameters are defined in the config file projects/CIFAR10Demo/scripts/config_train_resnet018_cifar.py. In order to run the demo code you will have to open the file and adjust some paths according to your system. Set the variable 'pytorchtools_path' to the location where you downloaded the pytorchtools to. Then, change the 'root' parameter to the location where you want the CIFAR10 dataset to be downloaded.

In Projects/CIFAR10Demo/scripts open a terminal and run
    bash job.sh
The command will train and test 5 networks with different seed on the CIFAR10 dataset. The results are exported into Projects/CIFAR10Demo/log. Each individual training process is exported to a different subdirectory. 

To evaluate the network, open and run the jupyter notebook Projects/CIFAR10Demo/scripts/evaluation.ipynb