import os
from collections import OrderedDict

def get_config(gpu=0, seed=42):

    batch_size = 512
    epochs = 100
    lr_init = batch_size / 64. * 1e-3
    lr_decay_start = 60
    n_tests = 20
    norm_mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
    norm_std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
    pytorchtools_path = '/nfshome/linse/Documents/development/pytorch_classification/pytorchtools'
    project_script_path = '/nfshome/linse/Documents/development/pytorch_classification/projects/CIFAR10Demo/scripts'

    config = {
        'log_path' : f'/nfshome/linse/Documents/development/pytorch_classification/projects/CIFAR10Demo/log/cifar10',
        'info' : "RestNet018 is trained on the official CIFAR10 train set and validated on the official CIFAR10 test set.",
        'seed' : seed,
        'epochs' : epochs,
        'num_workers' : 2,
        'pin_memory' : False,
        'no_cuda' : False,
        'cuda_device' : f'cuda:{gpu}',
        'save_data_paths' : True,
        'path_of_configfile' : os.path.abspath(__file__),
    }

    #networks
    config['networks'] = {}

    item = {}; config['networks']['network_main'] = item
    item['source'] = os.path.join(pytorchtools_path, 'ptnetworks/ResNetCIFAR.py')
    item['params'] = {
        'variant' : 'resnet018',
        'n_classes' : 10, 
        'pretrained' : False,
        }
    

    #loss functions
    config['loss_fns'] = {}

    item = {}; config['loss_fns']['loss_fn_main'] = item
    item['source'] = os.path.join(pytorchtools_path, 'ptutils/CrossEntropyLoss.py')
    item['params'] = {}


    #optimizers
    config['optimizers'] = {}

    item = {}; config['optimizers']['optimizer_main'] = item
    item['source'] = os.path.join(pytorchtools_path, 'ptutils/Lamb2.py')
    item['params'] = {
        'lr' : lr_init, 
        'betas' : (0.9, 0.999),
        'weight_decay' : 1.,
    }


    #transformations
    config['transforms'] = {}

    item = {}; config['transforms']['transform_train'] = item
    item['source'] = os.path.join(project_script_path, 'TransformCIFARTrainBaseline.py')
    item['params'] = {
        'norm_mean' :  norm_mean, 
        'norm_std' : norm_std,
        }
    
    item = {}; config['transforms']['transform_test'] = item
    item['source'] = os.path.join(project_script_path, 'TransformCIFARTest.py')
    item['params'] = {
        'norm_mean' :  norm_mean, 
        'norm_std' : norm_std,
        }


    #datasets
    config['datasets'] = {}

    item = {}; config['datasets']['dataset_train'] = item
    item['source'] = os.path.join(project_script_path, 'CIFAR10.py')
    item['params'] = {
        'transform' : 'transform_train',
        'root' : '/data/tmp_data',
        'download' : True,
        'b_train' : True,
    }

    item = {}; config['datasets']['dataset_test'] = item
    item['source'] = os.path.join(project_script_path, 'CIFAR10.py')
    item['params'] = {
        'transform' : 'transform_test',
        'root' : '/data/tmp_data',
        'download' : False,
        'b_train' : False,
    }


    #loaders
    config['loaders'] = {}

    item = {}; config['loaders']['loader_train'] = item
    item['source'] = os.path.join(pytorchtools_path, 'ptutils/DataloaderGetter.py')
    item['params'] = {
        'dataset' : 'dataset_train',
        'batch_size' : batch_size,
        'b_shuffled' : True,
    }

    item = {}; config['loaders']['loader_test'] = item
    item['source'] = os.path.join(pytorchtools_path, 'ptutils/DataloaderGetter.py')
    item['params'] = {
        'dataset' : 'dataset_test',
        'batch_size' : batch_size * 2,
        'b_shuffled' : False,
    }


    #scheduler modules
    config['scheduler_modules'] = OrderedDict()

    item = {}; config['scheduler_modules']['training_initializer'] = item
    item['source'] = os.path.join(pytorchtools_path, 'ptschedulers/SchedulerTrainingInitializationModule.py')
    item['params'] = {}


    item = {}; config['scheduler_modules']['scheduler_trainer'] = item
    item['source'] = os.path.join(pytorchtools_path, 'ptschedulers/SchedulerTrainingModule.py')
    item['params'] = {}


    item = {}; config['scheduler_modules']['scheduler_validator'] = item
    item['source'] = os.path.join(pytorchtools_path, 'ptschedulers/SchedulerValidationModule.py')
    item['params'] = {
        'active_epochs' : [round(i * config['epochs'] / n_tests) for i in range(1, n_tests)],
    }


    item = {}; config['scheduler_modules']['scheduler_model_saver'] = item
    item['source'] = os.path.join(pytorchtools_path, 'ptschedulers/SchedulerSaveNetModule.py')
    item['params'] = {}


    item = {}; config['scheduler_modules']['scheduler_lr'] = item
    item['source'] = os.path.join(pytorchtools_path, 'ptschedulers/SchedulerLearningRateModifierModule.py')
    item['params'] = {
        'schedule' : {
            lr_decay_start : lr_init * 3e-1, 
            lr_decay_start + int(5. / 10. * (config['epochs'] - lr_decay_start)) : lr_init * 1e-1, 
            lr_decay_start + int(6. / 10. * (config['epochs'] - lr_decay_start)) : lr_init * 3e-2, 
            lr_decay_start + int(7. / 10. * (config['epochs'] - lr_decay_start)) : lr_init * 1e-2, 
            lr_decay_start + int(8. / 10. * (config['epochs'] - lr_decay_start)) : lr_init * 3e-3, 
            lr_decay_start + int(9. / 10. * (config['epochs'] - lr_decay_start)) : lr_init * 1e-3,
        }
    }


    item = {}; config['scheduler_modules']['scheduler_save_output_final_test'] = item
    item['source'] = os.path.join(pytorchtools_path, 'ptschedulers/SchedulerLogOutputModule.py')
    item['params'] = {
        'loader' : 'loader_test',
        'keys_to_log_from_original_data_anyways' : ['label', 'id'],
        'filename' : 'outputs_test.pkl',
    }

    return config
    