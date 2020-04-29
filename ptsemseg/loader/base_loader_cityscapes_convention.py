import os
import torch
import numpy as np
import glob
import json

from PIL import Image
from torch.utils import data

from ptsemseg.loader.label_handler.label_handler import label_handler
from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale


def _get_json_param_value(img_path, json_param):

    dirname = os.path.dirname(img_path)
    img_basename = os.path.basename(img_path)
    img_splitname = os.path.splitext(img_basename)[0].split('_')

    json_path = os.path.join(
        dirname,
        'measurements_%05d_%s.json' \
        %(int(img_splitname[1]), img_splitname[2])
    )
    
    with open(json_path, 'r') as measurement_file:
        measurements = json.load(measurement_file)
    
    if json_param not in measurements.keys():
        print("Json param %s is not found in %s." %(json_param, json_path))
    
    return measurements[json_param]


class BaseLoaderCityscapesConvention(data.Dataset):
    
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
    
    def _init_classifier_head_labels(self, config, classifier_type):

        if classifier_type not in config: return (False, None)

        labels = {}
        for name, label in config[classifier_type].items():
            labels[name] = label
        
        return (True, labels)

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
        self.label_handler = label_handler(self.ignore_index)

        self.bin_classifiers, self.bin_label = self._init_classifier_head_labels(config, 'bin_classifiers')
        self.classifiers, self.classifier_label = self._init_classifier_head_labels(config, 'classifiers')

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
        
        name = img_path.split(os.sep)[-2:]
        name = os.path.join(name[0], name[1])

        img = Image.open(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = Image.open(lbl_path)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8), dataset_type)
        
        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl, index)

        lbl_dict = {"seg": lbl}
        # only append if used, only lbl_dict["seg"] is required
        self._append_bin_classifier_label(lbl_dict, img_path)
        self._append_classifier_label(lbl_dict, img_path)
        
        return img, lbl_dict, name

    def _append_bin_classifier_label(self, lbl_dict, img_path):
        
        if not self.bin_classifiers: return

        json_param_value = _get_json_param_value(img_path, 'path_type')

        for name, pos_labels in self.bin_label.items():
            if json_param_value in pos_labels:
                label = 1.0
            else:
                label = -1.0
            lbl_dict[name] = torch.from_numpy(np.array(label)).long()

    def _append_classifier_label(self, lbl_dict, img_path):
        
        if not self.classifiers: return
        
        json_param_value = _get_json_param_value(img_path, 'path_type')
        
        for name, pos_labels in self.classifier_label.items():
            
            if json_param_value not in pos_labels:
                print("[LOADER] ERROR! Unknown classification label %s at %s" %(json_param_value, img_path))
                return
            
            label = pos_labels.index(json_param_value)
            lbl_dict[name] = torch.from_numpy(np.array(label)).long()

    def transform(self, img, lbl, index):
        """transform

        :param img:
        :param lbl:
        """
        img = np.array(Image.fromarray(img).resize(
                (self.img_size[1], self.img_size[0])))  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)

        value_scale = 255
        mean = [0.406, 0.456, 0.485]
        mean = [item * value_scale for item in mean]
        std = [0.225, 0.224, 0.229]
        std = [item * value_scale for item in std]

        if self.img_norm:
            img = (img - mean) / std

        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

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

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

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

    # used in validation only in saving images, to convert back to test labels
    def decode_segmap_id(self, temp):
        return temp

    def encode_segmap(self, mask, dataset_type):
        # Put all void classes to zero
        if dataset_type == 'cityscapes':
            mask = self.label_handler.label_cityscapes(mask)
        if dataset_type == 'mapillary':
            mask = self.label_handler.label_mapillary(mask)
        if dataset_type == 'bdd100k':
            mask[mask==255] = self.ignore_index
        
        return mask


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
