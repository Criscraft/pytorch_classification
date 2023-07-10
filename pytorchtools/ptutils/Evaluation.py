import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
from torchvision import transforms
import pickle as pkl
import sklearn.manifold as manifold
import sklearn.metrics as skmetrics
from itertools import cycle

""" read and plot logged data """

def load_log_data(filename):
        data = None
        with open(filename, 'r') as f:
            try:
                data = pd.read_csv(f, sep='\t', index_col=False)
                if data['epoch'].iloc[-1] == -1:
                    data.drop(data.index[-1], inplace=True)
            except ValueError:
                print(filename + 'is missing')
        return data


def plot_training_history(filenames_train=[""], filenames_val=[""], list_to_plot=['loss'], exp_names=[], y_labels=[], y_lim=1e6, ncol=2, max_legend_size=5, log_scale=[], display_train_legend=False ,display_val_legend=True):
    nrow = np.ceil(len(list_to_plot) / float(ncol)).astype(int)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*6,nrow*6))
    if nrow > 1:
        axes = [a for sub_list in axes for a in sub_list]
    
    cmap = plt.get_cmap('Set1')
    colors = [cmap(x) for x in np.linspace(0.,1.,9) + 0.5/9.]
    color_cycle = cycle(colors)
    
    for exp_id, (filename_train, filename_val) in enumerate(zip(filenames_train, filenames_val)):
        
        #load train data
        if filename_train:
            plot_data_train = load_log_data(filename_train)
        else:
            plot_data_train = None
        
        #load val data if it exists
        if filename_val:
            plot_data_val = load_log_data(filename_val)
        else:
            plot_data_val = None
        
        color=next(color_cycle)
        
        for i, label in enumerate(list_to_plot):
            ax = axes[i]
            plt.sca(ax)

            #ax.set_title(get_latex_compatible_text(label))
            if not hasattr(ax, 'handle_list'):
                ax.handle_list = []
                ax.legend_texts = []
            
            #plot train data
            if plot_data_train is not None and label in plot_data_train:
                X = plot_data_train['epoch']
                y = plot_data_train[label].to_numpy()
                y[y>y_lim] = y_lim
                handle, = ax.plot(X, y, color=color, linestyle='solid')
                if exp_names and len(ax.handle_list) <= max_legend_size and display_train_legend:
                    ax.handle_list.append(handle)
                    text = exp_names[exp_id] + ' Training'
                    ax.legend_texts.append(get_latex_compatible_text(text))
            
            #plot val data
            if plot_data_val is not None and label in plot_data_val:
                X = plot_data_val['epoch']
                y = plot_data_val[label].to_numpy()
                y[y>y_lim] = y_lim
                handle, = ax.plot(X, y, color=color, linestyle='dashed')
                if exp_names and len(ax.handle_list) <= max_legend_size and display_val_legend:
                    ax.handle_list.append(handle)
                    text = exp_names[exp_id] + ' Validation'
                    ax.legend_texts.append(get_latex_compatible_text(text))
            
    for i in range(len(list_to_plot)):
        ax = axes[i]
        plt.sca(ax)
        plt.grid()
        if hasattr(ax, 'handle_list'):
            plt.legend(ax.handle_list, ax.legend_texts, fontsize=6) 
        if y_labels:
            ax.set_ylabel(y_labels[i], fontsize=12)
        ax.set_xlabel('epochs', fontsize=12)
        if log_scale and log_scale[i]:
            ax.set_yscale('log')
    
    fig.tight_layout()
    return fig


""" evaluate raw network outputs and evaluate """

def get_latex_compatible_text(text):
    text = text.replace('_', '\_')
    #text = text.encode('unicode')
    return text


def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


def load_outputs_targets(path_list):
    #loads network outputs and original target labels
    logits = []
    labels_orig = []
    
    for path in path_list:
        if path[-4:] != '.pkl':
            path = os.path.join(path, 'outputs_test.pkl')
        try:
            with open(path, "rb") as handle:
                output_dict = pkl.load(handle)
            logits.append(output_dict['logits'])
            labels_orig.append(output_dict['label_orig'])
        except FileNotFoundError:
            pass
    
    if logits:
        logits = torch.stack(logits, 0)
        labels_orig = torch.stack(labels_orig, 0)
        return logits, labels_orig
    else:
        return None, None

def get_loss_values(filename):
    logits, labels_orig = load_outputs_targets([filename])
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    losses = loss_fn(logits[0], labels_orig[0])
    return losses
    
def get_logits_and_original_labels(filename):
    logits, labels_orig = load_outputs_targets([filename])
    if logits is None:
        return None, None
    logits = logits[0] # we only read one single file
    labels_orig = labels_orig[0] # we only need one single file
    logits = F.softmax(logits, dim=1)
    return logits, labels_orig
    

def get_logits_and_original_labels_from_ensemble(filenames):
    logits, labels_orig = load_outputs_targets(filenames)
    if logits is None:
        return None, None
    #logits has shape n_models, n_images, n_classes
    logits = F.softmax(logits, dim=2)
    logits = logits.mean(dim=0)
    #logits has shape n_images, n_classes
    labels_orig = labels_orig[0] # we only need one single file
    return logits, labels_orig
    

""" Implement performance metrics """
def get_sensitivity(TP, TN, FP, FN):
    out = TP / (TP + FN + 1e-8)
    return out.numpy()
def get_specificity(TP, TN, FP, FN):
    out = TN / (TN + FP + 1e-8)
    return out.numpy()
def get_precision(TP, TN, FP, FN):
    out =  TP / (TP + FP + 1e-8)
    return out.numpy()
def get_f1_score(TP, TN, FP, FN):
    out = 2*TP/(2*TP + FP + FN + 1e-8)
    return out.numpy()
def get_MCC(TP, TN, FP, FN):
    out = (TP*TN - FP*FN) / (np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) + 1e-8)
    return out.numpy()


def get_TP_TN_FP_FN(labels_true, labels_pred, label_of_positive=0):
    label_of_positive = torch.tensor(label_of_positive, dtype=torch.long, requires_grad=False)
    TP = ((labels_pred==label_of_positive).bool() & (labels_true==label_of_positive).bool()).sum().float()
    TN = ((labels_pred!=label_of_positive).bool() & (labels_true!=label_of_positive).bool()).sum().float()
    FP = ((labels_pred==label_of_positive).bool() & (labels_true!=label_of_positive).bool()).sum().float()
    FN = ((labels_pred!=label_of_positive).bool() & (labels_true==label_of_positive).bool()).sum().float()
    return TP, TN, FP, FN
    
    
def get_accuracy(logits, labels_orig, labels_pred, label_of_positive=None):
    acc = (labels_pred == labels_orig.reshape_as(labels_pred)).float().mean()
    return acc.numpy()


def get_ROC_results(logits, labels_orig, labels_pred, label_of_positive=0):
    # returns array with rows: thresholds, sensitivities, specificities
    
    if logits is None:
        return None, None, None
    n_classes = len(labels_orig.unique())
    thresholds = np.linspace(0., 1., 51)
    
    if n_classes==2:
        labels_of_positive = [label_of_positive]
    else:
        labels_of_positive = list(range(n_classes))
    
    sens_list = []
    spec_list = []
    thre_list = []
    
    for label_of_positive in labels_of_positive:
        sens = []
        spec = []
        for threshold in thresholds:
            classified_as_positive = (logits[:,label_of_positive] >= threshold).bool()
            is_positive = (label_of_positive == labels_orig).bool()
            TP = (classified_as_positive & is_positive).sum().float()
            TN = (~classified_as_positive & ~is_positive).sum().float()
            FP = (classified_as_positive & ~is_positive).sum().float()
            FN = (~classified_as_positive & is_positive).sum().float()
            sens.append(get_sensitivity(TP, TN, FP, FN))
            spec.append(get_specificity(TP, TN, FP, FN))
        sens = np.array(sens)
        spec = np.array(spec)
        sens_list.append(sens)
        spec_list.append(spec)
        thre_list.append(thresholds)
        
    return np.array([thre_list, sens_list, spec_list])

def get_confusion_matrix(logits, labels_orig, labels_pred, label_of_positive=None):
    conf_mat = skmetrics.confusion_matrix(labels_orig, labels_pred)
    #conf_mat += 1e-12
    #conf_mat /= conf_mat.sum(axis=1)
    return conf_mat

def get_CMC_data(logits, labels_orig, labels_pred, label_of_positive):
    n_classes = len(np.unique(labels_orig))
    ranks_of_targets = np.zeros(n_classes)
    for logit, label_orig in zip(logits, labels_orig):
        ranking = np.argsort(-logit)
        ranks_of_targets = ranks_of_targets + (ranking == label_orig)
    cmc = np.cumsum(ranks_of_targets) / len(labels_orig)
    return cmc

""" Framework to compute metrics for arbitrary combination of experiments """
def get_class_metrics(logits, labels_orig, labels_pred, mode='acc', label_of_positive=None):
    function_dict = {
        'acc' : get_accuracy,
        'ROC' : get_ROC_results,
        'confusion_matrix' : get_confusion_matrix,
        'cmc' : get_CMC_data,
    }
    return function_dict[mode](logits, labels_orig, labels_pred, label_of_positive)


def get_n_class_metrics(labels_orig, labels_pred, mode='sens', label_of_positive=None):
    function_dict = {
        'sens' : get_sensitivity,
        'spec' : get_specificity,
        'pres' : get_precision,
        'f1sc' : get_f1_score,
        'mcc' : get_MCC,
    }
        
    n_classes = len(labels_orig.unique())
    if n_classes > 2:
        results = []
        for label in range(n_classes):
            TP, TN, FP, FN = get_TP_TN_FP_FN(labels_orig, labels_pred, label_of_positive=label)
            results.append(function_dict[mode](TP, TN, FP, FN))
        return np.stack(results)
    
    else:
        TP, TN, FP, FN = get_TP_TN_FP_FN(labels_orig, labels_pred, label_of_positive)
        return function_dict[mode](TP, TN, FP, FN)


def get_classification_metrics_from_multiple_sources(
    filenames,
    experiment_name, 
    class_metric_names=['acc', 'ROC', 'confusion_matrix'], 
    n_class_metric_names=['sens', 'spec', 'pres', 'f1sc'], 
    b_ensemble=False, 
    b_average_metrics=False, 
    label_of_positive=None, 
    labels=[]):
    
    """
    Outputs are two dictionaries single_exp_dict, combined_exp_dict. 
    Each contains evaluation metrics. 
    The format is dict(metric_name : ndarray(nexperiments, ...)) 
    where ... depends on the metric.
    nexperiments is 1 for combined_exp_dict
    Exeptions for the given format are the elements labels, labelstring, label_of_positive.
    """
    
    #create keys for output dict
    single_exp_dict = {}
    standard_dict = {}
    combined_exp_dict = {}
    
    single_exp_dict['id'] = []
    if labels:
        standard_dict['labels'] = labels
        standard_dict['labelstring'] = '_'.join([str(i) + ':' + label for i, label in enumerate(labels)])
    if not isinstance(label_of_positive, list):
        label_of_positive = [label_of_positive]
    if len(label_of_positive) != len(filenames):
        label_of_positive = [label_of_positive[0] for i in range(len(filenames))]
    standard_dict['label_of_positive'] = label_of_positive
    
    # compute metrics
    for filename, label_of_positive in zip(filenames, label_of_positive):
        
        single_exp_dict['id'].append(filename)
        logits, labels_orig = get_logits_and_original_labels(filename)
        if labels_orig is None:
            continue
            
        # compute class_metrics
        labels_pred = logits.argmax(1)
        for mode in class_metric_names:
            if not mode in single_exp_dict:
                single_exp_dict[mode] = []
            class_metrics = get_class_metrics(logits, labels_orig, labels_pred, mode=mode, label_of_positive=label_of_positive)
            single_exp_dict[mode].append(class_metrics)
        
        # compute n_class_metrics, these are evaluated for each class separatedly 
        for mode in n_class_metric_names:
            n_class_metrics = get_n_class_metrics(labels_orig, labels_pred, mode=mode, label_of_positive=label_of_positive)
            if n_class_metrics.ndim==0:
                if not mode in single_exp_dict:
                    single_exp_dict[mode] = []
                single_exp_dict[mode].append(n_class_metrics)
            else:
                if not (mode + '_macro_avg') in single_exp_dict:
                    single_exp_dict[mode + '_macro_avg'] = []
                    single_exp_dict[mode + '_n_class'] = []
                single_exp_dict[mode + '_macro_avg'].append(n_class_metrics.mean())
                single_exp_dict[mode + '_n_class'].append(n_class_metrics)
        
    # combine the previously validated models to an ensemble by averaging the posterior probabilitites
    if b_ensemble:
        
        logits, labels_orig = get_logits_and_original_labels_from_ensemble(filenames)
        labels_pred = logits.argmax(1)
        if labels_pred is not None:
            
            combined_exp_dict['id'] = [experiment_name]
            
            # compute class_metric_names
            for mode in class_metric_names:
                class_metrics = get_class_metrics(logits, labels_orig, labels_pred, mode=mode, label_of_positive=label_of_positive)
                combined_exp_dict[mode + '_ensemble'] = [class_metrics]
            
            # compute n_class_metric_names
            for mode in n_class_metric_names:
                n_class_metrics = get_n_class_metrics(labels_orig, labels_pred, mode=mode, label_of_positive=label_of_positive)
                if n_class_metrics.ndim==0:
                    combined_exp_dict[mode + '_ensemble'] = [n_class_metrics]
                else:
                    combined_exp_dict[mode + '_ensemble' + '_macro_avg'] = [n_class_metrics.mean()]
                    combined_exp_dict[mode + '_ensemble' + '_n_class'] = [n_class_metrics]
                    
    if single_exp_dict:
        single_exp_dict.update(standard_dict)
    if combined_exp_dict:
        combined_exp_dict.update(standard_dict)
        combined_exp_dict['label_of_positive'] = combined_exp_dict['label_of_positive'][0]
    
    single_exp_dict = {key : to_numpy(value) for (key, value) in single_exp_dict.items()}
    combined_exp_dict = {key : to_numpy(value) for (key, value) in combined_exp_dict.items()}
    
    # average performance metrics of single models
    if b_average_metrics:
        no_average = ['label_of_positive', 'labels', 'labelstring', 'id']
        combined_exp_dict['id'] = np.array([experiment_name])
        mean_dict = {key + '_mean' : np.array([metric.mean(0)]) for key, metric in single_exp_dict.items() if not key in no_average}
        std_dict = {key + '_std' : np.array([metric.std(0)]) for key, metric in single_exp_dict.items() if not key in no_average}
        combined_exp_dict.update(mean_dict)
        combined_exp_dict.update(std_dict)
            
    return single_exp_dict, combined_exp_dict


def to_numpy(value):
    return np.array(value)
    

def get_single_exp_df(metric_dict, n_class_metric_names):
    not_include = ["labels"]
    n_experiments = len(metric_dict['id'])
    
    # make all items have dim of n_experiments
    for key in list(metric_dict.keys()):
        if not metric_dict[key].shape:
            metric_dict[key] = np.array([metric_dict[key]])
        if len(metric_dict[key]) < n_experiments:
            metric_dict[key] = np.repeat(metric_dict[key], n_experiments)
            
    # create dictionary for dataframe construction
    sub_dict = {key : value for key, value in metric_dict.items() if (isinstance(value, np.ndarray) and value.ndim < 2 and len(value) == n_experiments)}
    sub_dict.update({mode + '_n_class_' + metric_dict['labels'][j] : metric_dict[mode + '_n_class'][:,j] for mode in n_class_metric_names for j in range(len(metric_dict['labels'])) if (mode + '_n_class') in metric_dict})
    
    for key in not_include:
        if key in sub_dict:
            del sub_dict[key]
    
    return pd.DataFrame(sub_dict, index=metric_dict['id'])
                  
    
def get_combined_exp_df(metric_dict, n_class_metric_names):
    not_include = ["labels"]
    
    # create dictionary for dataframe construction
    sub_dict = {key : value[0] for key, value in metric_dict.items() if value.ndim>0 and value[0].size==1}
    sub_dict.update({mode + '_ensemble_n_class_' + metric_dict['labels'][j] : metric_dict[mode + '_ensemble' + '_n_class'][0][j] for mode in n_class_metric_names for j in range(len(metric_dict['labels'])) if (mode + '_ensemble' + '_n_class') in metric_dict})
    sub_dict.update({mode + '_mean_n_class_' + metric_dict['labels'][j] : metric_dict[mode + '_mean' + '_n_class'][0][j] for mode in n_class_metric_names for j in range(len(metric_dict['labels'])) if (mode + '_mean' + '_n_class') in metric_dict})
    sub_dict.update({mode + '_std_n_class_' + metric_dict['labels'][j] : metric_dict[mode + '_std' + '_n_class'][0][j] for mode in n_class_metric_names for j in range(len(metric_dict['labels'])) if (mode + '_std' + '_n_class') in metric_dict})
    
    for key in not_include:
        if key in sub_dict:
            del sub_dict[key]
            
    return pd.DataFrame(sub_dict, index=[metric_dict['id']])
    
    
""" Plotting, accepts data structures created by functions above that compute performance metrics """ 
def draw_confusion_matrix(conf_mat, labels=[], label_of_positive=None, vmax=None, show_numbers=False, show_colorbar=True, figsize=(3.5, 3)):
    n_classes = len(conf_mat[0])
    
    if labels:
        assert(n_classes == len(labels)), "Labels do not fit number of classes! n_classes: " + str(n_classes) + " labels: " + ', '.join(labels)
    
    if label_of_positive is not None and label_of_positive > 0:
        #permute conf_mat such that the 'positive' class is at top left corner
        assert(labels)
        conf_mat = bring_to_topleft_corner(conf_mat, label_of_positive)
        labels[label_of_positive], labels[0] = labels[0], labels[label_of_positive] 
    
    colors = ["bisque", "darkorange"]
    fig, ax = plt.subplots(figsize=figsize)
    if vmax is None:
        vmax = conf_mat.max()
    myimage = ax.imshow(conf_mat, cmap=mpl.colors.LinearSegmentedColormap.from_list("mycmap", colors), norm=mpl.colors.Normalize(vmin=0, vmax=vmax, clip=True), origin='upper')
    if show_colorbar:
        fig.colorbar(myimage, ticks=np.linspace(0, vmax, 6))
    if show_numbers:
        for i in range(n_classes):
            for j in range(n_classes):
                plt.text(x=j, y=i, s=f"{conf_mat[i][j]:{1}}", ha='center', va='center', fontsize=11)
    ax.set_xticks(np.arange(n_classes))
    if labels:
        ax.set_xticklabels(labels, ha='center', va='center')
        ax.set_yticklabels(labels, rotation=90, ha='center', va='center')
    ax.set_yticks(np.arange(n_classes))
    ax.set_ylim(bottom=n_classes - 0.5, top=-0.5)
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    labeltop=True) # labels along the bottom edge are off
    plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,
    labelleft=True) # labels along the bottom edge are off
    ax.set(xlabel='predicted', ylabel='actual')
    fig.tight_layout()
    return fig, ax


def bring_to_topleft_corner(conf_mat, label_of_positive):
    if label_of_positive == 0:
        return conf_mat
    conf_mat = conf_mat.copy()
    conf_mat[np.array([0, label_of_positive])] = conf_mat[np.array([label_of_positive, 0])]
    conf_mat[:,np.array([0, label_of_positive])] = conf_mat[:,np.array([label_of_positive, 0])]
    return conf_mat
    
    
def plot_confusion_matrices(single_exp_metrics, combined_exp_metrics, vmax=500, result_path='../log', save_extension='pdf', save_results=True):
    
    for i, single_exp_metric in enumerate(single_exp_metrics):
        if 'confusion_matrix' in single_exp_metric:
            for cm, name in zip(single_exp_metric['confusion_matrix'], single_exp_metric['id']):
                print(name)
                fig, _ = draw_confusion_matrix(cm, labels=single_exp_metric.get('labels', []), vmax=vmax, show_numbers=True, show_colorbar=False)
                if save_results:
                    path = os.path.join(result_path, ("confusion_matrix_" + name.replace('/', '_') + "." + save_extension))
                    fig.savefig(path)
    
    for i, combined_exp_metric in enumerate(combined_exp_metrics):   
        if 'confusion_matrix_ensemble' in combined_exp_metric:
            print(combined_exp_metric["id"][0] + '_ensemble')
            fig, ax = draw_confusion_matrix(combined_exp_metric["confusion_matrix_ensemble"][0], labels=combined_exp_metric.get('labels', []), vmax=vmax, show_numbers=True, show_colorbar=False)
            if save_results:
                path = os.path.join(result_path, ("confusion_matrix_ensemble_" + combined_exp_metric["id"][0].replace('/', '_') + "." + save_extension))
                fig.savefig(path)
        if 'confusion_matrix_avg' in combined_exp_metric:
            print(combined_exp_metric["id"][0] + '_avg')
            fig, ax = draw_confusion_matrix(combined_exp_metric["confusion_matrix_avg"][0], labels=combined_exp_metric.get('labels', []), vmax=vmax, show_numbers=True, show_colorbar=False)
            if save_results:
                path = os.path.join(result_path, ("confusion_matrix_avg_" + combined_exp_metric["id"][0].replace('/', '_') + "." + save_extension))
                fig.savefig(path)


def draw_ROC_curve(ROC_array_list, experiment_labels=[''], class_labels=[], figsize=(3.5, 3)):
    #ROC_array_list is a list of ndarrays. Each ndarray contains thresholds, sens_arr and spec_arr.
    
    n_classes = len(ROC_array_list[0][1])
    assert len(experiment_labels) == len(ROC_array_list), "experiment_labels does not match the number of ROC items"
    
    if class_labels:
        assert(len(class_labels = n_classes))
    else:
        class_labels = ['' for _ in range(n_classes)]


    fig, ax = plt.subplots(figsize=figsize)
    ax.set(xlabel='1 - specificity', ylabel='sensitivity')
    plt.grid()
    cmap = plt.get_cmap('Set1')
    colors = [cmap(x) for x in np.linspace(0.,1.,9) + 0.5/9.]
    #remove green color to make plots friendly for color-blind people
    del colors[2]
    color_cycle = cycle(colors) 
    handles = []
    legend_texts = []
    
    if n_classes==2:
        for i in range(len(ROC_array_list)):
            sens_arr = ROC_array_list[i][1]
            spec_arr = ROC_array_list[i][2]
            #auc = sens_arr.mean()
            auc = np.abs(np.trapz(sens_arr[0], 1. - spec_arr[0]))
            text = experiment_labels[i] + f" AUC={auc:{3}.{3}}"
            handle, = ax.plot(1. - spec_arr[0], sens_arr[0], color=next(color_cycle))
            handles.append(handle)
            legend_texts.append(text)
    else:
        for i in range(len(ROC_array_list)):
            sens_arr = ROC_array_list[i][1]
            spec_arr = ROC_array_list[i][2]
            #auc = sens_arr.mean(1)
            auc = np.abs(np.trapz(sens_arr, 1. - spec_arr, axis=1))
            #sens_arr, spec_arr, auc are 2d arrays now
            for j in range(n_classes):
                text = experiment_labels[i] + ' ' + class_labels[j] + f", AUC={auc[j]:{3}.{3}}"
                handle, = ax.plot(1. - spec_arr[j], sens_arr[j], color=next(color_cycle))
                handles.append(handle)
                legend_texts.append(text)

    plt.legend(handles, legend_texts, fontsize=8)    
    fig.tight_layout()
    
    return fig, ax


def plot_ROC_curves(single_exp_metrics, combined_exp_metrics, result_path='../log', save_extension='pdf', save_results=True):
    
    for i, single_exp_metric in enumerate(single_exp_metrics):
        if 'ROC' in single_exp_metric:
            for roc, name in zip(single_exp_metric['ROC'], single_exp_metric['id']):
                print(name)
                fig, _ = ev.draw_ROC_curve([roc], class_labels=single_exp_metric.get('labels', []))
                if save_results:
                    path = os.path.join(result_path, ("ROC_" + name.replace('/', '_') + "." + save_extension))
                    fig.savefig(path)
    
    for i, combined_exp_metric in enumerate(combined_exp_metrics):   
        if 'ROC_ensemble' in combined_exp_metric:
            print(combined_exp_metric["id"][0] + '_ensemble')
            fig, _ = ev.draw_ROC_curve(combined_exp_metric['ROC_ensemble'], class_labels=combined_exp_metric.get('labels', []))
            if save_results:
                path = os.path.join(result_path, ("ROC_ensemble_" + combined_exp_metric["id"][0].replace('/', '_') + "." + save_extension))
                fig.savefig(path)
        if 'ROC_avg' in combined_exp_metric:
            print(combined_exp_metric["id"][0] + '_avg')
            fig, _ = ev.draw_ROC_curve(combined_exp_metric['ROC_avg'], class_labels=combined_exp_metric.get('labels', []))
            if save_results:
                path = os.path.join(result_path, ("ROC_avg_" + combined_exp_metric["id"][0].replace('/', '_') + "." + save_extension))
                fig.savefig(path)


transform_to_pil = transforms.ToPILImage()
def tensor_to_pil(image):
    target_mean = np.array([0.485, 0.456, 0.406]).reshape((3,1,1))
    target_std = np.array([0.229, 0.224, 0.225]).reshape((3,1,1))

    image = image.numpy().squeeze()
    if image.ndim == 3:
        image = image / image.std((1,2), keepdims=True) * target_std
        image = image - image.mean((1,2), keepdims=True) + target_mean
        image = image.transpose((1,2,0))
    image[image < 0.] = 0.
    image[image > 1.] = 1.
    image = image * 255
    image = np.round(image)
    image = image.astype(np.uint8)
    image = transform_to_pil(image)
    return image

def plot_embedding(X, y):
    fig, ax = plt.subplots(figsize=(7, 7))
    markers = ['.', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']
    marker_cycle = cycle(markers) 
    unique = np.unique(y)
    for i, unique_class in enumerate(unique):
        mask = y==unique_class
        X_plot = X[mask]
        y_plot = y[mask]
        plt.scatter(X_plot[:, 0], X_plot[:, 1], marker=next(marker_cycle), s=12)#, color=plt.get_cmap('rainbow')(i / len(unique)))

    #plt.xticks([]), plt.yticks([])
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    return fig, ax