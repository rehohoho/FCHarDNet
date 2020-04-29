import os
import yaml
import numpy as np
from PIL import Image
import argparse

from torch.utils import data

from ptsemseg.loader import get_loader
from ptsemseg.augmentations import get_composed_augmentations, get_composed_augmentations_softmax
from ptsemseg.utils import get_cityscapes_image_from_tensor


def save_batch_noreplace(path_array, data_array, mask = False):

    assert len(path_array) == len(data_array), "Unable to save batch. Number of path names %s and number of items %s different." %(len(path_array), len(data_array))
    
    for path, data in zip(path_array, data_array):

        dirname = os.path.dirname(path) # create directory if does not exist
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        i = 1
        while os.path.exists(path):     # get name_idx if path is duplicate
            path = os.path.splitext(path)
            path = "%s_%d%s" %(path[0], i, path[1])
            i += 1
            print(path)

        if path.endswith(".npy"):       # file type depends on path
            np.save(path, data.astype(np.float16))
        else:
            assert data.shape[2] == 3
            get_cityscapes_image_from_tensor(data, mask=mask).save(path)


class DataLoader():

    def __init__(self, cfg, dataset_split, save_paths):
        
        self.dataloader_type = cfg["data"]["dataset"]

        self.cfg = cfg
        self.dataset_split = dataset_split
        self._init_augmentations()
        self._init_dataloader()

        self.save_paths = save_paths

    def _init_augmentations(self):
        
        augmentations = self.cfg["training"].get("augmentations", None)
        if self.dataloader_type == "softmax_cityscapes_convention":
            self.data_aug = get_composed_augmentations_softmax(augmentations)
        else:
            self.data_aug = get_composed_augmentations(augmentations)

    def _init_dataloader(self):
        
        cfg_data = self.cfg["data"]
        cfg_training = self.cfg["training"]

        data_loader = get_loader(cfg_data["dataset"])
        data_path = cfg_data["path"]

        self.t_loader = data_loader(
            data_path,
            config = cfg_data,
            is_transform=True,
            split=cfg_data[self.dataset_split],
            img_size=(cfg_data["img_rows"], cfg_data["img_cols"]),
            augmentations=self.data_aug,
        )
        
        n_classes = self.t_loader.n_classes
        self.train_loader = data.DataLoader(
            self.t_loader,
            batch_size=cfg_training["batch_size"],
            num_workers=cfg_training["n_workers"],
            shuffle=True,
        )

    def get_save_path_names(self, name_array, save_path_param, file_type):
        
        return [
            os.path.join(self.save_paths[save_path_param], i) \
                .replace(".png", file_type)
            for i in name_array
        ]

    def save_augmented_images(self, vis = False):
        """ Saves different files depending on vis as follows
        vis == True
            - rgb png file
            - visualised mask png file
            - visualised argmaxed softmax png file
            - visualised ignore mask png file
        vis == False
            - rgb npy file
            - mask npy file
            - softmax npy file
        """

        for (images, label_dict, name) in self.train_loader:
            
            print(name)

            if "softmax" in label_dict:
                softmax_array = label_dict["softmax"].data.cpu().numpy()

            image_array = images.data.cpu().numpy()
            label_array = label_dict["seg"].data.cpu().numpy()
            
            if vis:
                file_type = ".png"
                
                image_array = image_array.transpose(0, 2, 3, 1)
                label_array = [self.t_loader.decode_segmap(i) for i in label_array]

                if "softmax" in label_dict:
                    softmax_array = np.argmax(softmax_array, axis = 1)
                    softmax_array = [self.t_loader.decode_segmap(i) for i in softmax_array]
                    
                    ignore_mask_array = self.t_loader.extract_ignore_mask(images).data.cpu().numpy()
                    ignore_mask_array = [self.t_loader.decode_segmap(i) for i in ignore_mask_array]
                    ignore_path = self.get_save_path_names(name, "label", file_type)
                    ignore_path = [i.replace("seg", "ignore") for i in ignore_path]
                    save_batch_noreplace(ignore_path, ignore_mask_array, mask=True)

            else:
                file_type = ".npy"
            
            img_path = self.get_save_path_names(name, "image", file_type)
            lbl_path = self.get_save_path_names(name, "label", file_type)
            save_batch_noreplace(img_path, image_array)
            save_batch_noreplace(lbl_path, label_array, mask=True)
            
            if "softmax" in label_dict:
                sftmax_path = self.get_save_path_names(name, "softmax", file_type)
                save_batch_noreplace(sftmax_path, softmax_array, mask=True)
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/softmax.yml",
        help="Configuration file to use",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train_split",
        help="train_split or val_split",
    )
    parser.add_argument(
        "--repeat",
        default=0,
        type=int,
        help="number of times to repeat runing through data loader",
    )
    parser.add_argument(
        "--vis",
        default=False,
        action="store_true",
        help="visualise mask or just save as npy",
    )
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)
    
    data_config = cfg["data"]
    root = os.path.join(data_config["path"], data_config["version"])
    save_paths_list = [
        os.path.join(root, "%s_aug" %i, cfg["data"][args.dataset_split])
        for i in os.listdir(root)
    ]
    save_paths = {
        "image": [i for i in save_paths_list if "image" in i][0],
        "label": [i for i in save_paths_list if "seg" in i][0],
        "softmax": [i for i in save_paths_list if "logits" in i][0]
    }

    data_loader = DataLoader(cfg, args.dataset_split, save_paths)
    for run in range(args.repeat + 1):
        data_loader.save_augmented_images(vis = args.vis)
