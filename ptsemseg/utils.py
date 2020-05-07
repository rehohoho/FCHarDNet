"""
Misc Utility functions
"""
import os
import logging
import datetime
import numpy as np
from PIL import Image
import json
from collections import OrderedDict


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def alpha_blend(input_image, segmentation_mask, alpha=0.5):
    """Alpha Blending utility to overlay RGB masks on RBG images
        :param input_image is a np.ndarray with 3 channels
        :param segmentation_mask is a np.ndarray with 3 channels
        :param alpha is a float value
    """
    blended = np.zeros(input_image.size, dtype=np.float32)
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    return blended


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_logger(logdir):
    logger = logging.getLogger("ptsemseg")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def get_cityscapes_image_from_tensor(image, mask = False, get_image_obj = True):
    """
    Reverse image transformation
        - normalisation for image
        - /=255 for mask

        :param image is a np.array
    """
    if mask:
        image *= 255
    else:
        std = np.array([57.375, 57.12 , 58.395])
        mean = np.array([103.53 , 116.28 , 123.675])
        image = (image * std + mean)
        image = image[:, :, ::-1]
    
    image = image.astype(np.uint8)
    if get_image_obj:
        image = Image.fromarray(image)
    
    return image


def get_json_param_value(img_path, json_param):

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


def get_sampling_weights(dataset, sampling_cfg):
    """ Get sampling weights for dataset based on label_name
    Sampler weights do not have to add up to 1

    Currently sampling strategy only supports balancing of one classiffication metric
    """
    assert len(sampling_cfg) == 1, "Sampling function only supports one metric. Metrics: %s" %sampling_cfg.keys()
    tar_label_name = list(sampling_cfg.keys())[0]
    
    if tar_label_name == "seg":
        # TODO data balancing based on segmentation mask
        print("Data balancing does not support segmentation mask yet.")
        return

    if tar_label_name in dataset.bin_label.keys():
        pos_label = dataset.bin_label[tar_label_name]
    elif tar_label_name in dataset.classifier_label.keys():
        pos_label = dataset.classifier_label[tar_label_name]
    else:
        print("Target label %s is not found in %s or %s." %(tar_label_name, dataset.bin_label.keys(), dataset.classifier_label.keys()))
        print("Data sampling balancing only supported for classification labels.")
        return

    files = dataset.files[dataset.split]
    labels = []

    for img_path in files:
        json_param_value = get_json_param_value(img_path.rstrip(), 'path_type')
        labels.append(pos_label.index(json_param_value))

    class_weights = sampling_cfg[tar_label_name]/np.bincount(labels) # number of samples of each idx
    return class_weights[labels]