from PIL import Image
import imgaug.augmenters as iaa
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import numpy as np


class ComposeSM(object):
    """
    Require softmax layer to be np.ndarray of (HWC)
    Require image layer to be np.ndarray of (HWC)
    """
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, segmap, softmax):
        
        segmap = SegmentationMapsOnImage(segmap, shape=segmap.shape)
        softmax = HeatmapsOnImage(softmax, shape = softmax.shape)

        assert img.shape[:2] == segmap.shape[:2] == softmax.shape[:2]
        for a in self.augmentations:
            img, segmap, softmax = a(img, segmap, softmax)
        
        return img, segmap.get_arr(), softmax.get_arr()
        

class BaseImgaug(object):

    def __call__(self, img, segmap, softmax):
        
        return self.seq(image = img, segmentation_maps = segmap, heatmaps = softmax)
        

class RandomHorizontallyFlipSM(BaseImgaug):
    
    def __init__(self, p):
        
        self.seq = iaa.Sequential([
                iaa.Fliplr(p),
        ])


class RandomScaleCropSM(BaseImgaug):
    
    def __init__(self, size):
        
        self.seq = iaa.Sequential(
            [
                iaa.Resize((0.5, 2.0)), # random scale
                iaa.PadToFixedSize(width=size[0], height=size[1]),
                iaa.CropToFixedSize(width=size[0], height=size[1]),
            ], random_order = False)