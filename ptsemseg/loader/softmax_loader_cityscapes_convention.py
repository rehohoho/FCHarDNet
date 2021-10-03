import os
import torch
import numpy as np
from scipy.special import softmax
import glob

from PIL import Image
from torch.utils import data

from ptsemseg.loader.label_handler.label_handler import label_handler
import imgaug.augmenters as iaa
from imgaug.augmentables.heatmaps import HeatmapsOnImage


class SoftmaxLoaderCityscapesConvention(data.Dataset):
    """
    Requires dataset to have subdirectories: *images and *seg and *softmax_temp
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

    mean_rgb = {
        "pascal": [103.939, 116.779, 123.68],
        "cityscapes": [0.0, 0.0, 0.0],
    }  # pascal mean for PSPNet and ICNet pre-trained model

    image_suffix = {
        'cityscapes': ('_leftImg8bit.png', '_gtFine_labelIds.png'),
        'mapillary': ('.jpg', '.png'),
        'bdd100k': ('.jpg', '_train_id.png'),
        'scooter': ('.png', '.png'),
        'detector': ('.png', '.png')
    } # to search for image / label / softmax files

    def _init_get_files(self, datasets):
        self.files[self.split] = []

        for dataset_type in datasets:
            file_suffix = [v for k, v in self.image_suffix.items() if k in dataset_type]
            assert len(file_suffix) == 1, 'Dataset type not specified properly. %s found.' %[k for k in self.image_suffix.keys() if k in dataset_type]

            self.files[self.split] += [ file for file in glob.glob(
                os.path.join(self.root, '%s/*images' %dataset_type, self.split, '**/*'+file_suffix[0][0]),
                recursive = True
            )]
    
    def __init__(
        self,
        root,
        config,
        split="train",
        is_transform=False,
        img_size=(1024, 2048),
        augmentations=None,
        img_norm=True,
        test_mode=False,
    ):
        self.root = root
        self.split = split

        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 19

        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array(self.mean_rgb["cityscapes"])
        self.files = {}

        assert 'version' in config
        datasets = config['version'].replace(' ', '').split(',')
        print('Datasets used:', datasets)
        
        self._init_get_files(datasets)
        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.root))
        print("Found %d %s images" % (len(self.files[split]), split))

        self.ignore_index = 250
        assert 'softmax_temperature' in config
        self.softmax_temperature = config['softmax_temperature']
        # self.label_handler = label_handler(self.ignore_index)

        self.softmax_resize_seq = iaa.Sequential([ # used at transform, to resize to img_size
            iaa.Resize({"height": img_size[0], "width": img_size[1]})
        ])

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def _get_corresponding_label(self, img_path, dataset_type, subfolder_name):
        
        suffix_replace = [v for k, v in self.image_suffix.items() if k in dataset_type]
        assert len(suffix_replace) == 1, 'Dataset type not specified properly. %s found.' %[k for k in self.image_suffix.keys() if k in dataset_type]
        lbl_path = img_path.replace(suffix_replace[0][0], suffix_replace[0][1])
        lbl_path = lbl_path.replace('images', subfolder_name)

        return lbl_path
        
    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        dataset_type = img_path.split(self.root)[-1].split(os.sep)[0]
        lbl_path = self._get_corresponding_label(img_path, dataset_type, 'seg')
        lbl_seg_path = self._get_corresponding_label(img_path, dataset_type, 'logits').replace('.png', '.npy')
            
        name = img_path.split(os.sep)[-2:]
        name = os.path.join(name[0], name[1])

        img = np.array(Image.open(img_path), dtype=np.uint8) #np.ndarray of (HWC)
        lbl = Image.open(lbl_path)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8), dataset_type)
        lbl_seg = np.load(lbl_seg_path).transpose(1,2,0).astype(np.float32)
        lbl_seg = softmax(lbl_seg/self.softmax_temperature, axis = 2)

        if self.augmentations is not None:
            img, lbl, lbl_seg = self.augmentations(img, lbl, lbl_seg)

        if self.is_transform:
            img, lbl, lbl_seg = self.transform(img, lbl, lbl_seg, index) # normalisation, transposing, convert to tensor

        lbl_dict = {
            "seg": lbl,
            "softmax": lbl_seg
        }

        return img, lbl_dict, name

    def transform(self, img, lbl, lbl_seg, index):
        """ perform standardisation and resizing """

        img = np.array(Image.fromarray(img).resize(
                (self.img_size[1], self.img_size[0])))  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        if self.img_norm:
            img = (img - [103.53, 116.28, 123.675]) / [57.375, 57.120000000000005, 58.395]
        img = img.transpose(2, 0, 1) # NHWC -> NCHW

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = np.array(Image.fromarray(lbl).resize(
                (self.img_size[1], self.img_size[0]), resample=Image.NEAREST))
        lbl = lbl.astype(int)
        
        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            print(self.n_classes, index)
            raise ValueError("Segmentation map contained invalid class values")

        lbl_seg = HeatmapsOnImage(lbl_seg, shape = lbl_seg.shape)
        lbl_seg = self.softmax_resize_seq(heatmaps = lbl_seg).get_arr()
        lbl_seg = lbl_seg.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        lbl_seg = torch.from_numpy(lbl_seg).float()

        return img, lbl, lbl_seg
    
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

    def encode_segmap(self, mask, dataset_type):
        # Put all void classes to zero
        if dataset_type == 'cityscapes':
            mask = self.label_handler.label_cityscapes(mask)
        if dataset_type == 'mapillary':
            mask = self.label_handler.label_mapillary(mask)
        if dataset_type == 'bdd100k':
            mask[mask==255] = self.ignore_index
        
        return mask

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
        return temp


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
