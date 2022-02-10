import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
from torchvision import transforms
import ptutils.PytorchHelpers as ph


def check_data_mean_std(dataset):
    """
    Dataset has to output RGB images.
    """

    aggregate = (0., 0., 0.)

    for i in range(len(dataset)):
        image = dataset[i]['data'].numpy().mean(axis=(1,2))
        aggregate = update_mean_std(aggregate, image)

    image_means, image_stds, _ = finalize_mean_std(aggregate)

    return image_means, image_stds


def update_mean_std(existingAggregate, newValue):
    """Welford's Online algorithm for computing mean and std of a distribution online.
    mean accumulates the mean of the entire dataset.
    m2 aggregates the squared distance from the mean.
    count aggregates the number of samples seen so far.

    Arguments:
        existingAggregate {tuple} -- Intermediate results (count, mean, m2)
        newValue {float} -- A new value drawn from the distribution

    Returns:
        tuple -- updated aggregate (count, mean, m2)
    """

    (count, mean, m2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    m2 += delta * delta2

    return (count, mean, m2)

def finalize_mean_std(existingAggregate):
    """Retrieve the mean, variance and sample variance from an aggregate

    Arguments:
        existingAggregate {tuple} -- Intermediate results (count, mean, m2)
        
    Returns:
        tuple -- distribution statistics: (mean, standard deviation, standard deviation with sample normalization
    """

    (count, mean, m2) = existingAggregate
    (mean, variance, sample_variance) = (mean, m2/count, m2/(count - 1)) 
    if count < 2:
        return float('nan')
    else:
        return (mean, np.sqrt(variance), np.sqrt(sample_variance))

def plot_img_sizes(dataset, filename=''):

    shapes = []

    for i in range(len(dataset)):
        image = dataset[i]['data'].numpy()
        shapes.append(image.shape[1:]) #first dim: height, second dim:width
    shapes = np.array(shapes).transpose()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    nbins=200
    k = kde.gaussian_kde([shapes[0],shapes[1]])
    xi, yi = np.mgrid[shapes[0].min():shapes[0].max():nbins*1j, shapes[1].min():shapes[1].max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    zi = zi.reshape(xi.shape)
    # Make the plot
    theCM = plt.cm.get_cmap('Greys')
    theCM._init()
    alphas = np.linspace(0., 1.0, theCM.N)
    theCM._lut[:-3,-1] = alphas
    myplot = plt.pcolormesh(xi, yi, zi, cmap=theCM)
    cbar = fig.colorbar(myplot)
    cbar.set_label('Density')
    cbar.ax.set_yticklabels([])
    
    scatterplot = ax.scatter(
        shapes[0],
        shapes[1],
        marker=',',
        s=1,
        color='black')
    
    myplot = plt.pcolormesh(xi, yi, zi, cmap=plt.cm.Greys)
    
    ax.set_xlabel('Image Height (Pixels)', fontsize=12)
    ax.set_ylabel('Image Width (Pixels)', fontsize=12)
    plt.grid()
    fig.tight_layout()
    if filename:
        fig.savefig(filename, transparent=False, dpi=80, bbox_inches="tight")

    print('mean height')
    print(np.mean(shapes[0]))
    print('mean width')
    print(np.mean(shapes[1]))
    print('mean_aspect_ratio')
    print(np.mean(shapes[0]/shapes[1]))


def plot_class_distributions(dataloader, filename=''):
    labels = []
    for batch in dataloader:
        labels.extend(batch['label'])
    labels = np.array(labels)

    print('number of samples: {}'.format(len(labels)))
    unique = np.unique(labels)
    n_classes = len(unique)
    print('n_classes: ' + str(n_classes))

    hist, bins = np.histogram(labels, n_classes)
    print('mean class count')
    print(np.mean(hist))
    print('min_class_count')
    print(np.min(hist))
    print('max_class_count')
    print(np.max(hist))
    #sort_inds = np.argsort(hist)
    #hist = hist[sort_inds]
    
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(range(n_classes), hist)
    ax.set_xlabel('class', fontsize=12)
    ax.set_ylabel('number of images', fontsize=12)
    fig.tight_layout()
    if filename:
        fig.savefig(filename, transparent=False, dpi=80, bbox_inches="tight")
    return fig


def check_if_dataloaders_disjunkt(dataloader1, dataloader2):
    img_names_1 = []
    for batch in dataloader1:
        img_names_1.extend(batch['path'])
    img_names_1 = np.array(img_names_1)

    img_names_2 = []
    for batch in dataloader2:
        img_names_2.extend(batch['path'])
    img_names_2 = np.array(img_names_2)

    print(np.intersect1d(img_names_1, img_names_2).shape)


def plot_pixel_distribution(dataloader, filename=''):
    data = []
    for batch in dataloader:
        data.append(batch['data'])
    data = ph.concat(data)
    fig = plot_histogram(data, axis=1, nbins=50)
    if filename:
        fig.savefig(filename, transparent=False, dpi=80, bbox_inches="tight")


def plot_histogram(x, axis=0, nbins=10):
    if not isinstance(x, np.ndarray):
        x = x.numpy()
    n_plots = x.shape[axis]
    ncol = 3
    nrow = np.ceil(n_plots / float(ncol)).astype(int)
    fig, _ = plt.subplots(nrow, ncol, figsize=(ncol*3.2,nrow*3.2))
    for i, x_slice in enumerate(np.moveaxis(x, axis, 0)):
        ax = plt.subplot(nrow, ncol, i+1)
        minimum = x_slice.min()
        maximum = x_slice.max()
        hist, bins = np.histogram(x_slice.reshape((-1)), nbins, (minimum, maximum))
        ax.bar(bins[:-1], hist, (bins[-1]-bins[0])/nbins, align='edge', color='b')
    fig.tight_layout()
    return fig


transform_to_pil = transforms.ToPILImage()


def tensor_to_pil(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    target_mean = np.array(mean).reshape((3,1,1))
    target_std = np.array(std).reshape((3,1,1))
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


def plot_image_batch(dataloader, batch_index=0, filename='', norm_mean=[], norm_std=[]):
    for i, batch in enumerate(dataloader):
        if i == batch_index:
            break
    
    ncol = 4
    nrow = np.ceil(batch['data'].shape[0] / float(ncol)).astype(int)
    fig, _ = plt.subplots(nrow, ncol, figsize=(ncol*2.5,nrow*2.5))
    for i, (image_tensor, label) in enumerate(zip(batch['data'], batch['label'])):
        ax = plt.subplot(nrow, ncol, i+1)
        ax.imshow(tensor_to_pil(image_tensor, norm_mean, norm_std))
        ax.set_title(str(label))
        plt.axis('off')
    fig.tight_layout()
    
    if filename:
        fig.savefig(filename, transparent=False, dpi=80, bbox_inches="tight")

    return fig