# Codebase for image classification in Pytorch

This repository contains a classification pipeline for Pytorch. The directory 'pytorchtools' contains scripts for data analysis, network implementation, training loop and evaluation. The directory 'projects' contains specific experiments. In the demo project ResNet018 is trained and tested on CIFAR10.

## Setup

The required python packages can be found in the file requirements.txt
You can install them with 
    pip install -r requirements.txt 
Then, add pytorchtools directory to your PYTHONPATH.

## Get started with the demo

In Projects/CIFAR10Demo/scripts open a terminal and run
    bash job.sh
The command will train and test 5 networks with different seed on the CIFAR10 dataset and it will use the GPU with index 0 for this task. The results are written into Projects/CIFAR10Demo/log. Each individual training process is documented within an own subdirectory.
To evaluate the network, open and run the jupyter notebook Projects/CIFAR10Demo/scripts/evaluation.ipynb