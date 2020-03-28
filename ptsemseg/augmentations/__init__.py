import logging
from ptsemseg.augmentations.augmentations import (
    AdjustContrast,
    AdjustGamma,
    AdjustBrightness,
    AdjustSaturation,
    AdjustHue,
    RandomCrop,
    RandomHorizontallyFlip,
    RandomVerticallyFlip,
    Scale,
    RandomScaleCrop,
    RandomSized,
    RandomSizedCrop,
    RandomRotate,
    RandomTranslate,
    CenterCrop,
    Compose,
)
from ptsemseg.augmentations.augmentations_imgaug import (
    RandomHorizontallyFlipSM,
    RandomScaleCropSM,
    ComposeSM,
)

logger = logging.getLogger("ptsemseg")


def get_composed_augmentations(aug_dict):
    
    key2aug = {
        "gamma": AdjustGamma,
        "hue": AdjustHue,
        "brightness": AdjustBrightness,
        "saturation": AdjustSaturation,
        "contrast": AdjustContrast,
        "rcrop": RandomCrop,
        "hflip": RandomHorizontallyFlip,
        "vflip": RandomVerticallyFlip,
        "scale": Scale,
        "rscale_crop": RandomScaleCrop,
        "rsize": RandomSized,
        "rsizecrop": RandomSizedCrop,
        "rotate": RandomRotate,
        "translate": RandomTranslate,
        "ccrop": CenterCrop,
    }

    if aug_dict is None:
        logger.info("Using No Augmentations")
        return None

    augmentations = []
    for aug_key, aug_param in aug_dict.items():
        augmentations.append(key2aug[aug_key](aug_param))
        logger.info("Using {} aug with params {}".format(aug_key, aug_param))
    return Compose(augmentations)


def get_composed_augmentations_softmax(aug_dict):
    
    key2aug = {
        "hflip": RandomHorizontallyFlipSM,
        "rscale_crop": RandomScaleCropSM,
    }
    
    if aug_dict is None:
        logger.info("Using No Augmentations")
        return None

    augmentations = []
    for aug_key, aug_param in aug_dict.items():
        augmentations.append(key2aug[aug_key](aug_param))
        logger.info("Using {} aug with params {}".format(aug_key, aug_param))
    return ComposeSM(augmentations)