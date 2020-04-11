import logging
import functools
import torch

from ptsemseg.loss.loss import (
    cross_entropy2d,
    bootstrapped_cross_entropy2d,
    soft_and_hard_target_cross_entropy,
    multi_scale_cross_entropy2d,
    l1
)

logger = logging.getLogger("ptsemseg")

key2loss = {
    "cross_entropy": cross_entropy2d,
    "bootstrapped_cross_entropy": bootstrapped_cross_entropy2d,
    "soft_and_hard_target_cross_entropy": soft_and_hard_target_cross_entropy,
    "multi_scale_cross_entropy": multi_scale_cross_entropy2d,
    "l1": l1
}


def get_loss_function(cfg, device=None):
    if cfg["training"]["loss"] is None:
        logger.info("Using default cross entropy loss")
        loss_fn = cross_entropy2d

    else:
        loss_dict = cfg["training"]["loss"]
        loss_name = loss_dict["name"]
        loss_params = {k: v for k, v in loss_dict.items() if k != "name"}

        if loss_name not in key2loss:
            raise NotImplementedError("Loss {} not implemented".format(loss_name))

        logger.info("Using {} with {} params".format(loss_name, loss_params))
        loss_fn = functools.partial(key2loss[loss_name], **loss_params)
    
    if "detector_loss" in cfg["training"]:

        loss_dict = cfg["training"]["detector_loss"]
        loss_name = loss_dict["name"]
        loss_params = {k: v for k, v in loss_dict.items() if k != "name"}
        
        if loss_name not in key2loss:
            raise NotImplementedError("Loss {} not implemented".format(loss_name))
        if device is not None:
            for param in loss_params.keys():
                if isinstance(loss_params[param], list):
                    loss_params[param] = torch.Tensor(loss_params[param]).to(device)

        logger.info("Using {} with {} params".format(loss_name, loss_params))
        loss_fn = (loss_fn, functools.partial(key2loss[loss_name], **loss_params) )

    return loss_fn
