from ptsemseg.augmentations import get_composed_augmentations_softmax
from PIL import Image
import numpy as np

CITYSCAPES_COLMAP = np.array( [
    [128, 64,128],    #road
    [244, 35,232],    #sidewalk
    [ 70, 70, 70],    #building
    [102,102,156],    #wall
    [190,153,153],    #fence
    [153,153,153],    #pole
    [250,170, 30],    #traffic light
    [220,220,  0],    #traffic sign
    [107,142, 35],    #vegetation
    [152,251,152],    #terrain
    [ 70,130,180],    #sky
    [220, 20, 60],
    [255,  0,  0],
    [  0,  0,142],
    [  0,  0, 70],
    [  0, 60,100],
    [  0, 80,100],
    [  0,  0,230],
    [119, 11, 32]
    
    ], dtype = np.uint8)

aug_dict = {
    'hflip': 0.5,
    'rscale_crop': [1024, 1024]
}

aug = get_composed_augmentations_softmax(aug_dict)
png = 'D:/perception_datasets/scooter_halflabelled/scooter_images/train/11-May-2019-17-54-07/frame_0_mid.png'
seg = 'D:/perception_datasets/scooter_halflabelled/scooter_seg/train/11-May-2019-17-54-07/frame_0_mid.png'
npy = 'D:/perception_datasets/scooter_halflabelled/scooter_softmax_temp/train/11-May-2019-17-54-07/frame_0_mid.npy'
im = np.array(Image.open(png))
seg = np.array(Image.open(seg))
npy = np.load(npy).astype(np.float32).transpose(1,2,0)

aug_im, aug_seg, aug_segtemp = aug(im, seg, npy)
aug_seg_im = CITYSCAPES_COLMAP[aug_seg]
aug_segtemp_im = CITYSCAPES_COLMAP[np.argmax(aug_segtemp, axis = 2).astype(np.uint8)]
Image.fromarray(aug_im).save('im_aug.png')
Image.fromarray(aug_seg_im).save('seg_aug.png')
Image.fromarray(aug_segtemp_im).save('npy_aug.png')