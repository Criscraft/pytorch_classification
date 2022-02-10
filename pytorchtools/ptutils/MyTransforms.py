import numpy as np
import numpy.random as random
from torchvision import transforms
from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
import torch
from torch.nn import functional as F
#from Networks.GaussianSmoothing import GaussianSmoothing


class MyRandomCrop(object):
    """Crop the given PIL Image at a random location."""

    def __init__(self, crop_ratio=0.05, b_keep_aspect_ratio=True):
        self.crop_ratio = crop_ratio
        self.b_keep_aspect_ratio = b_keep_aspect_ratio

    @staticmethod
    def get_params(img, crop_ratio, b_keep_aspect_ratio):
        """Get parameters for ``crop`` for a random crop.

        """
        w, h = img.size
        # i is the left pixel of the cropbox
        # j is the top pixel of the cropbox

        i = random.randint(0, h * crop_ratio)
        h_new = h - i - random.randint(0, h * crop_ratio)

        j = random.randint(0, w * crop_ratio)

        if b_keep_aspect_ratio:
            w_new = int(w * (h_new / h))
        else:
            w_new = w - j - random.randint(0, w * crop_ratio)

        return i, j, h_new, w_new

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """

        i, j, h_new, w_new = self.get_params(img, self.crop_ratio, self.b_keep_aspect_ratio)

        return img.crop((j, i, j + w_new, i + h_new))

    def __repr__(self):
        return self.__class__.__name__ + '(crop_ratio={0})'.format(self.crop_ratio)


class MyCenterCrop(object):
    """Crop the given PIL Image at a random location."""

    def __init__(self, crop_ratio=0.05):
        self.crop_ratio = crop_ratio

    @staticmethod
    def get_params(img, crop_ratio):
        """Get parameters for ``crop`` for a random crop.
        """

        w, h = img.size
        w_new = random.randint(0, w * 2 * crop_ratio)
        h_new = random.randint(0, h * 2 * crop_ratio)
        return (h_new, w_new)

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """

        output_size = self.get_params(img, self.crop_ratio)

        return transforms.functional.center_crop(img, output_size)

    def __repr__(self):
        return self.__class__.__name__ + '(crop_ratio={0})'.format(self.crop_ratio)



class PadToSize(object):
    """
    Adds padding in order to reach a desired image size
    """

    def __init__(self, image_size, fill):
        self.image_size = image_size
        self.fill = fill

    @staticmethod
    def get_params(img, image_size):
        """Get parameters for ``crop`` for a random crop.
        """
        w, h = img.size
        delta_height = image_size[0] - h
        delta_width = image_size[1] - w
        p_left = random.randint(0,delta_width+1)
        p_right = delta_width - p_left
        p_top = random.randint(0,delta_height+1)
        p_bottom = delta_height - p_top
        padding = (p_left, p_top, p_right, p_bottom)
        return padding

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        padding = self.get_params(img, self.image_size)
        return transforms.functional.pad(img, padding, fill=self.fill)

    def __repr__(self):
        return self.__class__.__name__ + '(crop_ratio={0})'.format(self.image_size)


class RandomScaleWithMaxSize(object):
    """
    Adds padding in order to reach a desired image size
    """

    def __init__(self, max_size, min_coverage):
        self.max_size = max_size
        self.min_coverage = min_coverage

    @staticmethod
    def get_params(img, max_size, min_coverage):
        """Get parameters for ``crop`` for a random crop.
        """
        w, h = img.size
        img_aspect_ratio = h / w
        allowed_aspect_ratio = max_size[0] / max_size[1]

        if img_aspect_ratio >= allowed_aspect_ratio:
            # h is comparably large and is the axis to control scaling
            max_scalefactor = max_size[0] / h
        else:
            # w is comparably large and is the axis to control scaling
            max_scalefactor = max_size[1] / w
        
        min_scalefactor = np.sqrt(min_coverage*max_size[0]*max_size[1]/h/w)

        if min_scalefactor > max_scalefactor:
            # img has an extreme aspect ratio and the min_coverage constraint cannot be fulfilled
            scalefactor = max_scalefactor
        else:
            scalefactor = min_scalefactor + random.random() * (max_scalefactor - min_scalefactor)
        
        h_new = int(h*scalefactor)
        w_new = int(w*scalefactor)

        return (h_new, w_new)

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        desired_size = self.get_params(img, self.max_size, self.min_coverage)
        return transforms.functional.resize(img, desired_size)

    def __repr__(self):
        return self.__class__.__name__ + '(max_size={0}, min_coverage={1})'.format(self.max_size, self.min_coverage)

class RandomSqueeze(object):
    """
    Squeeze image to target size times a random factor in the range of variation_factor. Try to keep number of pixels constant
    """
    def __init__(self, target_pixels, variation_factor):
        self.target_pixels = target_pixels
        self.variation_factor = variation_factor

    @staticmethod
    def get_params(img, target_pixels, variation_factor):
        w, h = img.size
        v = ((random.random() - 0.5) * 2 * variation_factor) + 1.
        
        h_new = int(np.sqrt(target_pixels * h * v / w))
        w_new = int(np.sqrt(target_pixels * w / h / v))

        return (h_new, w_new)

    def __call__(self, img):
        desired_size = self.get_params(img, self.target_pixels, self.variation_factor)
        return transforms.functional.resize(img, desired_size)

    def __repr__(self):
        return self.__class__.__name__ + '(target_pixels={0}, variation_factor={1})'.format(self.target_pixels, self.variation_factor)


class AddGaussianNoise(object):
    """
    Adds Gaussian noise
    """

    def __init__(self, blend_alpha_range):
        self.blend_alpha_range = blend_alpha_range

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """

        w, h = img.size
        if img.mode == 'RGB':
            c = 3
        elif img.mode == 'L':
            c = 1
        else:
            raise ValueError('Got image of unknown mode')
        noise_image = np.random.normal(0., 1., (h, w, c))
        noise_image = Image.fromarray(noise_image, img.mode)
        blend_alpha = np.random.random()*(self.blend_alpha_range[1] - self.blend_alpha_range[0]) + self.blend_alpha_range[0]

        return Image.blend(img, noise_image, blend_alpha)

    def __repr__(self):
        return self.__class__.__name__ + '(blend_alpha={0})'.format(self.blend_alpha_range)

class HistogramEqualization(object):
    """
    Equalize the image histogram
    """

    def __init__(self):
        pass

    def __call__(self, img):
        return ImageOps.equalize(img)

    def __repr__(self):
        return self.__class__.__name__


class GaussianBlur(object):
    """
    Adds Gaussian noise
    """

    def __init__(self, p=0.5, max_radius=3):
        self.p = p
        self.max_radius = max_radius

    def __call__(self, img):
        if np.random.random() > self.p:
            return img
        else:
            if self.max_radius == 2:
                radius = 2
            else:
                radius = np.random.randint(2, self.max_radius)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))

    def __repr__(self):
        return self.__class__.__name__ + '(p={0}, max_radius={1})'.format(self.p, self.max_radius)

    
class NormalizeToMeanStd(object):
    """Normalize a tensor image to have a given mean and standard deviation.
    """

    def __init__(self, target_mean, target_std, clip_zero=True, clip_one=False):
        self.target_mean = torch.FloatTensor(target_mean).reshape((3,1,1))
        self.target_std = torch.FloatTensor(target_std).reshape((3,1,1))
        self.clip_zero = clip_zero
        self.clip_one = clip_one

    def __call__(self, tensor):
        self.target_mean = self.target_mean.to(tensor.device)
        self.target_std = self.target_std.to(tensor.device)
        tensor = tensor / tensor.std(dim=2, keepdim=True).std(dim=1, keepdim=True) * self.target_std
        tensor = tensor - tensor.mean(dim=2, keepdim=True).mean(dim=1, keepdim=True) + self.target_mean
        if self.clip_zero: 
            tensor[tensor < 0.] = 0.
        if self.clip_one:
            tensor[tensor > 1.] = 1.
        return tensor


class ToPilImageCustom(object):
    """Normalize a tensor image to have a given mean and standard deviation.
    """

    def __init__(self, target_mean, target_std=None):
        self.target_mean = np.array(target_mean).reshape((3,1,1))
        self.target_std = np.array(target_std).reshape((3,1,1))

    def __call__(self, npimg):
        if isinstance(npimg, torch.Tensor):
            npimg = npimg.cpu().numpy()
        
        #npimg = npimg - npimg.min()
        #npimg = npimg / npimg.max()

        npimg = npimg / npimg.std((1,2), keepdims=True) * self.target_std
        npimg = npimg - npimg.mean((1,2), keepdims=True) + self.target_mean

        npimg[npimg < 0.] = 0.
        npimg[npimg > 1.] = 1.
        npimg = npimg * 255
        npimg = np.round(npimg)
        npimg = npimg.astype(np.uint8)
        npimg = np.transpose(npimg, (1, 2, 0))
        image = Image.fromarray(npimg, mode='RGB')
        return image

class L2Decay(object):
    """
    """

    def __init__(self, theta):
        self.theta = torch.FloatTensor([theta]).reshape((1,1,1))
        self.ones = torch.ones((1,1,1))

    def __call__(self, tensor):
        self.ones = self.ones.to(tensor.device)
        self.theta = self.theta.to(tensor.device)
        return tensor * (self.ones - self.theta)

class NormalizeToRange(object):
    """Normalize a tensor image to have a given mean and standard deviation.
    """

    def __init__(self, range=[[-1, -1, -1], [1, 1, 1]]):
        self.range = torch.FloatTensor(range).reshape((2,3,1,1))

    def __call__(self, tensor):
        self.range = self.range.to(tensor.device)
        tensor = tensor - tensor.min()
        tensor = tensor / tensor.max() * (self.range[1] - self.range[0])
        tensor = tensor + self.range[0]
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + ''

class GaussianBlurForTensors(object):

    def __init__(self, kernel_size=3, sigma=1., interval=1):
        self.interval = interval
        self.gauss_layer = GaussianSmoothing(channels=3, kernel_size=kernel_size, sigma=sigma, dim=2)
        self.epoch = 1
        self.kernel_size = kernel_size

    def __call__(self, tensor):
        if self.epoch % self.interval == 0:
            self.gauss_layer = self.gauss_layer.to(tensor.device)
            tensor = tensor.unsqueeze(0)
            tensor = self.gauss_layer(tensor)
            tensor = tensor.squeeze(0)
        self.epoch = self.epoch + 1
        return tensor

class ScaleUp(object):

    def __init__(self, interval=2, target_size=(500,500), rate_per_epoch=1):
        self.interval = interval
        self.epoch = 1
        self.target_size = np.array(target_size)
        self.rate_per_epoch = rate_per_epoch
        self.initial_size = [0, 0]

    def __call__(self, tensor):
        if self.epoch == 1:
            self.initial_size = np.array(tensor.shape[1:])
        if self.epoch % self.interval == 0:
            h = tensor.shape[1]
            w = tensor.shape[2]
            h_target = int(min(self.epoch * self.rate_per_epoch + self.initial_size[0], self.target_size[0]))
            w_target = int(min(self.epoch * self.rate_per_epoch + self.initial_size[1], self.target_size[1]))
            if h < h_target or w < w_target:
                tensor = tensor.unsqueeze(0)
                tensor = F.interpolate(tensor, size=(h_target, w_target), mode='bilinear')
                tensor = tensor.squeeze(0)
        self.epoch = self.epoch + 1
        return tensor

class ClipPixelsWithSmallNorm(object):

    def __init__(self, percentile=0.01, interval=1):
        self.interval = interval
        self.epoch = 1
        self.percentile = percentile

    def __call__(self, tensor):
        if self.epoch % self.interval == 0:
            norms = tensor.norm(dim=0, keepdim=True)
            threshold = np.abs(np.percentile(norms.cpu().numpy(), self.percentile))
            tensor[norms < tensor.new_full((3,1,1), threshold)] = tensor.min()

        self.epoch = self.epoch + 1
        return tensor


class RegularizeExtremePixels(object):

    def __init__(self, percentile_range=(5, 95), interval=1):
        self.interval = interval
        self.epoch = 1
        self.percentile_range = percentile_range

    def __call__(self, tensor):
        if self.epoch % self.interval == 0:
            npimg = tensor.cpu().numpy()
            threshold = np.percentile(npimg, self.percentile_range[1])
            npimg[npimg > threshold] = threshold

            threshold = np.percentile(npimg, self.percentile_range[0])
            npimg[npimg < threshold] = threshold

            tensor = torch.from_numpy(npimg).to(tensor.device)

        self.epoch = self.epoch + 1
        return tensor

class ShuffleImage(object):
    
    def __init__(self):
        pass

    def __call__(self, image):
        w, h = image.size
        image_px = image.getdata()
        image_px = np.reshape(image_px, (h, w, 3))
        
        x_coordinates = [1 * image_px.shape[2] // 4, 3 * image_px.shape[2] // 4]
        y_coordinates = [1 * image_px.shape[1] // 4, 3 * image_px.shape[1] // 4]
        seed = \
            image_px[y_coordinates[0], x_coordinates[0], :].sum() + \
            image_px[y_coordinates[0], x_coordinates[1], :].sum() + \
            image_px[y_coordinates[1], x_coordinates[0], :].sum() + \
            image_px[y_coordinates[1], x_coordinates[1], :].sum()

        randomstate = np.random.RandomState(int(seed))
        
        image_px = np.reshape(image_px, (h * w, 3))
        randomstate.shuffle(image_px)

        image_px = np.reshape(image_px, (h, w, 3))
        image = Image.fromarray(image_px.astype('uint8'))

        return image