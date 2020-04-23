import logging
import functools
import torch

from ptsemseg.loss.loss import (
    cross_entropy2d,
    bootstrapped_cross_entropy2d,
    soft_and_hard_target_cross_entropy,
    multi_scale_cross_entropy2d,
    l1,
    cross_entropy1d
)

logger = logging.getLogger("ptsemseg")

key2loss = {
    "cross_entropy": cross_entropy2d,
    "bootstrapped_cross_entropy": bootstrapped_cross_entropy2d,
    "soft_and_hard_target_cross_entropy": soft_and_hard_target_cross_entropy,
    "multi_scale_cross_entropy": multi_scale_cross_entropy2d,
    "l1": l1,
    "cross_entropy1d": cross_entropy1d
}


def get_loss_function(cfg, device=None):
    
    def _get_loss(loss_config):
        
        loss_dict = cfg["training"][loss_config]
        loss_name = loss_dict["name"]
        loss_params = {k: v for k, v in loss_dict.items() if k != "name"}
        
        if loss_name not in key2loss:
            raise NotImplementedError("Loss {} not implemented".format(loss_name))
        
        if device is not None:
            for param in loss_params.keys():
                if isinstance(loss_params[param], list):
                    loss_params[param] = torch.Tensor(loss_params[param]).to(device)
        
        logger.info("[LOSS] {}: Using {} with {} params".format(loss_config, loss_name, loss_params))
        
        return functools.partial(key2loss[loss_name], **loss_params)

    loss_dict = {}

    for key in cfg["training"]:
        if "loss" in key:
            loss_dict[key] = _get_loss(key)
    
    if len(loss_dict) == 0:
        logger.info(" No loss defined. Using default cross entropy loss")
        loss_dict["loss"] = cross_entropy2d

    return loss_dict
