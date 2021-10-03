import os
import torch
import numpy as np
import glob

from PIL import Image
from torch.utils import data

from ptsemseg.loader.label_handler.label_handler import label_handler
import imgaug.augmenters as iaa
from imgaug.augmentables.heatmaps import HeatmapsOnImage


class SoftmaxLoaderDirectLoad(data.Dataset):
    """
    Requires dataset to have subdirectories: *images and *seg and *softmax_temp
    Expects dataset generated from save_augmented_images.py

    - image, mask, softmax layer should have same base names
    - labels should be encoded from test to train, according to config used to get dataset
    """

    colors = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    def _init_get_files(self, datasets):
        
        self.files[self.split] = []

        for dataset_type in datasets:
            self.files[self.split] += [ file for file in glob.glob(
                os.path.join(self.root, '%s/*images' %dataset_type, self.split, '**/*.npy'),
                recursive = True
            )]

    def __init__(
        self,
        root,
        config,
        split="train",
        is_transform=False,     # not used
        img_size=(1024, 2048),  # not used
        augmentations=None,     # not used
        test_mode=False,
    ):
        print("[LOADER] Direct loader is used. No standardisation, transform, and augmentations applied.")
        print("[LOADER] This loader expects dataset generated from save_augmented_images.py")

        self.root = root
        self.split = split
        self.n_classes = 19
        self.files = {}

        assert 'version' in config
        datasets = config['version'].replace(' ', '').split(',')
        print('Datasets used:', datasets)

        self._init_get_files(datasets)
        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.root))
        print("Found %d %s images" % (len(self.files[split]), split))
        
        self.ignore_index = 250
        # self.label_handler = label_handler(self.ignore_index)

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        lbl_path = img_path.replace('images', 'seg') # all datasets have same basenames
        lbl_seg_path = img_path.replace('images', 'softmax_temp')
        
        name = img_path.split(os.sep)[-2:]
        name = os.path.join(name[0], name[1])

        img = np.load(img_path)
        lbl = np.load(lbl_path) # labels expected to be encoded, by config used to generate dataset
        lbl_seg = np.load(lbl_seg_path)

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            print(self.n_classes, index)
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        lbl_seg = torch.from_numpy(lbl_seg).float()

        lbl_dict = {
            "seg": lbl,
            "softmax": lbl_seg
        }

        return img, lbl_dict, name
    
    def extract_ignore_mask(self, image, device=None):
        """ Retrieve loss weights to zero padded portions of image/predictions 
        weight is repeated to fit shape of input / target in BCELoss
        
        Args:   image           NCHW tensor
        Return: ignore_mask     NHW tensor
        """

        std = torch.Tensor([57.375, 57.120000000000005, 58.395])
        mean = torch.Tensor([103.53, 116.28, 123.675])
        if device is not None:
            std = std.to(device)
            mean = mean.to(device)

        image_hwc = image.transpose(1,2).transpose(2,3)
        image_hwc = image_hwc * std + mean
        image_hwc = image_hwc.type(torch.cuda.LongTensor)

        padded_single_channel = torch.sum(image_hwc, dim=3)

        ignore_mask = torch.where(
            padded_single_channel == 0,
            padded_single_channel,
            torch.ones_like(padded_single_channel)
        )
        
        return ignore_mask

    # used for logging or testing in __main__ only, fixed to cityscapes convention
    def decode_segmap(self, temp):
        """ visualise segmentation map
        Args:       temp    HW nd.array
        Returns:    rgb     HWC nd.array
        """
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb
    
    # used in validation only in saving images
    def decode_segmap_id(self, temp):
        ids = np.zeros((temp.shape[0], temp.shape[1]),dtype=np.uint8)
        for l in range(0, self.n_classes):
            ids[temp == l] = self.valid_classes[l]
        return ids


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    augmentations = Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip(0.5)])

    local_path = '/home/whizz/Desktop/deeplabv3/datasets/'
    dst = scooterLoader(local_path, is_transform=True, augmentations=augmentations)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for (imgs, labels, _) in trainloader:
        
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = input()
        if a == "ex":
            break
        else:
            plt.close()
