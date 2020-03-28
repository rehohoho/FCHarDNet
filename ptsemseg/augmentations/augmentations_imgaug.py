from PIL import Image
import imgaug.augmenters as iaa
from imgaug.augmentables.heatmaps import HeatmapsOnImage
import numpy as np


class ComposeSM(object):
    """
    Require softmax layer to be np.ndarray of (HWC)
    Require image layer to be np.ndarray of (HWC)
    """
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, softmax):
        
        softmax = HeatmapsOnImage(softmax, shape = softmax.shape)

        assert img.shape[:2] == softmax.shape[:2]
        for a in self.augmentations:
            img, softmax = a(img, softmax)
        
        return img, softmax.get_arr()
        

class RandomHorizontallyFlipSM(object):
    def __init__(self, p):
        
        self.seq = iaa.Sequential([
                iaa.Fliplr(p),
        ])

    def __call__(self, img, softmax):
        
        return self.seq(image = img, heatmaps = softmax)


class RandomScaleCropSM(object):
    def __init__(self, size):
        
        self.seq = iaa.Sequential(
            [
                iaa.Resize((0.5, 2.0)), # random scale
                iaa.PadToFixedSize(width=size[0], height=size[1]),
                iaa.CropToFixedSize(width=size[0], height=size[1]),
            ], random_order = False)

            
    def __call__(self, img, softmax):

        return self.seq(image = img, heatmaps = softmax)