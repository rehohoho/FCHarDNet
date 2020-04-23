import os
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from math import log10, floor
from PIL import Image

from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader
from ptsemseg.utils import get_logger, get_cityscapes_image_from_tensor
from ptsemseg.metrics import runningScoreSeg, runningScoreClassifier, averageMeter
from ptsemseg.augmentations import get_composed_augmentations, get_composed_augmentations_softmax
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer

from tensorboardX import SummaryWriter


def write_images_to_board(loader, image, gt, pred, step, name, softmax_gt = None):
    
    batch_size = min(image.shape[0], 20)

    for i in range(batch_size):

        vis_gt = loader.decode_segmap(gt[i]) #takes HW nd.array, outputs HWC
        vis_gt = vis_gt.transpose(2, 0, 1) #change to CHW
        vis_gt = torch.Tensor(vis_gt).type('torch.cuda.FloatTensor')
        
        vis_pred = loader.decode_segmap(pred[i])
        vis_pred = vis_pred.transpose(2, 0, 1)
        vis_pred = torch.Tensor(vis_pred).type('torch.cuda.FloatTensor')

        writer.add_image('%s_%s_Image' %(name, i), image[i], step)
        writer.add_image('%s_%s_Label' %(name, i), vis_gt, step)
        writer.add_image('%s_%s_Pred' %(name, i), vis_pred, step)
        
        if softmax_gt is None:
            continue
        
        vis_softmax = loader.decode_segmap(softmax_gt[i]) #takes HW nd.array, outputs HWC
        vis_softmax = vis_softmax.transpose(2, 0, 1) #change to CHW
        vis_softmax = torch.Tensor(vis_softmax).type('torch.cuda.FloatTensor')
        writer.add_image('%s_%s_Softmax' %(name, i), vis_softmax, step)


def write_images_to_dir(loader, image, gt, pred, step, save_dir, name, softmax_gt = None):
    
    writer_label = [loader.decode_segmap(i) for i in gt]
    vis_pred = [ loader.decode_segmap(i) for i in pred]
    if softmax_gt is not None:
        writer_softmax = [loader.decode_segmap(i) for i in softmax_gt]
    
    save_path = os.path.join(save_dir, str(step))
    batch_size = min(image.shape[0], 20)

    for i in range(batch_size):
        get_cityscapes_image_from_tensor(image[i]).save('%s_%s_%d_Image.png' %(save_path, name, i))
        get_cityscapes_image_from_tensor(writer_label[i], mask=True).save('%s_%s_%d_Label.png' %(save_path, name, i))
        get_cityscapes_image_from_tensor(vis_pred[i], mask=True).save('%s_%s_%d_Pred.png' %(save_path, name, i))

        if softmax_gt is not None:
            get_cityscapes_image_from_tensor(writer_softmax[i], mask=True).save('%s_%s_%d_Softmax.png' %(save_path, name, i))


def compute_loss(loss_dict, images, label_dict, output_dict, device, t_loader):
    """ compute loss based on available labels and output types """

    loss = 0 # adding loss is a tensor function that can be backpropped

    for loss_name, loss_fn in loss_dict.items(): # only adds loss if it is defined
        
        k = loss_name.replace("_loss", "")
        if loss_name == "softmax_loss":
            loss += loss_fn(input = output_dict["seg"], hard_target = label_dict["seg"].to(device), 
                soft_target = label_dict["softmax"].to(device), 
                ignore_mask = t_loader.extract_ignore_mask(images).to(device)
            )
        else:
            loss += loss_fn(input = output_dict[k], target = label_dict[k].to(device))

    return loss


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)


def train(cfg, writer, logger, start_iter=0, model_only=False, gpu=-1, save_dir=None):

    # Setup seeds and config
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))
    
    # Setup device
    if gpu == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:%d" %gpu if torch.cuda.is_available() else "cpu")

    # Setup Augmentations
    augmentations = cfg["training"].get("augmentations", None)
    if cfg["data"]["dataset"] == "softmax_cityscapes_convention":
        data_aug = get_composed_augmentations_softmax(augmentations)
    else:
        data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]
    if "version" in cfg["data"]:
        version = cfg["data"]["version"]
    else:
        version = "cityscapes"

    t_loader = data_loader(
        data_path,
        config = cfg["data"],
        is_transform=True,
        split=cfg["data"]["train_split"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        augmentations=data_aug,
    )

    v_loader = data_loader(
        data_path,
        config = cfg["data"],
        is_transform=True,
        split=cfg["data"]["val_split"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
    )

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(
        t_loader,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"],
        shuffle=True,
    )

    valloader = data.DataLoader(
        v_loader, batch_size=cfg["training"]["batch_size"], num_workers=cfg["training"]["n_workers"]
    )

    # Setup Metrics
    running_metrics_val = {"seg": runningScoreSeg(n_classes)}
    if "classifiers" in cfg["data"]:
        for name, classes in cfg["data"]["classifiers"].items():
            running_metrics_val[name] = runningScoreClassifier( len(classes) )
    if "bin_classifiers" in cfg["data"]:
        for name, classes in cfg["data"]["bin_classifiers"].items():
            running_metrics_val[name] = runningScoreClassifier(2)

    # Setup Model
    model = get_model(cfg["model"], n_classes).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print( 'Parameters:',total_params )

    if gpu == -1:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    else:
        model = torch.nn.DataParallel(model, device_ids=[gpu])
    
    model.apply(weights_init)
    pretrained_path='weights/hardnet_petite_base.pth'
    weights = torch.load(pretrained_path)
    model.module.base.load_state_dict(weights)

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    print("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg["training"]["lr_schedule"])
    loss_dict = get_loss_function(cfg, device)

    if cfg["training"]["resume"] is not None:
        if os.path.isfile(cfg["training"]["resume"]):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
            )
            checkpoint = torch.load(cfg["training"]["resume"], map_location=device)
            model.load_state_dict(checkpoint["model_state"], strict=False)
            if not model_only:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                scheduler.load_state_dict(checkpoint["scheduler_state"])
                start_iter = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg["training"]["resume"], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg["training"]["resume"]))

    if cfg["training"]["finetune"] is not None:
        if os.path.isfile(cfg["training"]["finetune"]):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["finetune"])
            )
            checkpoint = torch.load(cfg["training"]["finetune"])
            model.load_state_dict(checkpoint["model_state"])

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100.0
    i = start_iter
    flag = True
    loss_all = 0
    loss_n = 0

    while i <= cfg["training"]["train_iters"] and flag:
        for (images, label_dict, _) in trainloader:
            i += 1
            start_ts = time.time()
            scheduler.step()
            model.train()

            images = images.to(device)
            optimizer.zero_grad()
            output_dict = model(images)

            loss = compute_loss(    # considers key names in loss_dict and output_dict
                loss_dict, images, label_dict, output_dict, device, t_loader
            )
            
            loss.backward()         # backprops sum of loss tensors, frozen components will have no grad_fn
            optimizer.step()
            c_lr = scheduler.get_lr()

            if i%1000 == 0:             # log images, seg ground truths, predictions
                pred_array = output_dict["seg"].data.max(1)[1].cpu().numpy()
                gt_array = label_dict["seg"].data.cpu().numpy()
                softmax_gt_array = None
                if "softmax" in label_dict:
                    softmax_gt_array = label_dict["softmax"].data.max(1)[1].cpu().numpy()
                write_images_to_board(t_loader, images, gt_array, pred_array, i, name = 'train', softmax_gt = softmax_gt_array)

                if save_dir is not None:
                    image_array = images.data.cpu().numpy().transpose(0, 2, 3, 1)
                    write_images_to_dir(t_loader, image_array, gt_array, pred_array, i, save_dir, name = 'train', softmax_gt = softmax_gt_array)

            time_meter.update(time.time() - start_ts)
            loss_all += loss.item()
            loss_n += 1
            
            if (i + 1) % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}  lr={:.6f}"
                print_str = fmt_str.format(
                    i + 1,
                    cfg["training"]["train_iters"],
                    loss_all / loss_n,
                    time_meter.avg / cfg["training"]["batch_size"],
                    c_lr[0],
                )

                print(print_str)
                logger.info(print_str)
                writer.add_scalar("loss/train_loss", loss.item(), i + 1)
                time_meter.reset()

            if (i + 1) % cfg["training"]["val_interval"] == 0 or (i + 1) == cfg["training"][
                "train_iters"
            ]:
                torch.cuda.empty_cache()
                model.eval()
                loss_all = 0
                loss_n = 0
                with torch.no_grad():
                    for i_val, (images_val, label_dict_val, _) in tqdm(enumerate(valloader)):
                        
                        images_val = images_val.to(device)
                        output_dict = model(images_val)
                        
                        val_loss = compute_loss(
                            loss_dict, images_val, label_dict_val, output_dict, device, v_loader
                        )
                        val_loss_meter.update(val_loss.item())

                        for name, metrics in running_metrics_val.items():
                            gt_array = label_dict_val[name].data.cpu().numpy()
                            if name+'_loss' in cfg['training'] and cfg['training'][name+'_loss']['name'] == 'l1':
                                pred_array = output_dict[name].data.cpu().numpy()
                                pred_array = np.sign(pred_array)
                                pred_array[pred_array == -1] = 0
                                gt_array[gt_array == -1] = 0
                            else:
                                pred_array = output_dict[name].data.max(1)[1].cpu().numpy()

                            metrics.update(gt_array, pred_array)

                softmax_gt_array = None # log validation images
                pred_array = output_dict["seg"].data.max(1)[1].cpu().numpy()
                gt_array = label_dict_val["seg"].data.cpu().numpy()
                if "softmax" in label_dict_val:
                    softmax_gt_array = label_dict_val["softmax"].data.max(1)[1].cpu().numpy()
                write_images_to_board(v_loader, images_val, gt_array, pred_array, i, 'validation', softmax_gt = softmax_gt_array)
                if save_dir is not None:
                    images_val = images_val.cpu().numpy().transpose(0, 2, 3, 1)
                    write_images_to_dir(v_loader, images_val, gt_array, pred_array, i, save_dir, name='validation', softmax_gt = softmax_gt_array)

                logger.info("Iter %d Val Loss: %.4f" % (i + 1, val_loss_meter.avg))
                writer.add_scalar("loss/val_loss", val_loss_meter.avg, i + 1)

                for name, metrics in running_metrics_val.items():
                    
                    overall, classwise = metrics.get_scores()
                    
                    for k, v in overall.items():
                        logger.info("{}_{}: {}".format(name, k, v))
                        writer.add_scalar("val_metrics/{}_{}".format(name, k), v, i + 1)

                        if k == cfg["training"]["save_metric"]:
                            curr_performance = v

                    for metric_name, metric in classwise.items():
                        for k, v in metric.items():
                            logger.info("{}_{}_{}: {}".format(name, metric_name, k, v))
                            writer.add_scalar("val_metrics/{}_{}_{}".format(name, metric_name, k), v, i + 1)

                    metrics.reset()
                
                state = {
                      "epoch": i + 1,
                      "model_state": model.state_dict(),
                      "optimizer_state": optimizer.state_dict(),
                      "scheduler_state": scheduler.state_dict(),
                }
                save_path = os.path.join(
                    writer.file_writer.get_logdir(),
                    "{}_{}_checkpoint.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                )
                torch.save(state, save_path)

                if curr_performance >= best_iou:
                    best_iou = curr_performance
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(
                        writer.file_writer.get_logdir(),
                        "{}_{}_best_model.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                    )
                    torch.save(state, save_path)
                torch.cuda.empty_cache()

            if (i + 1) == cfg["training"]["train_iters"]:
                flag = False
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/scooter.yml",
        help="Configuration file to use",
    )
    parser.add_argument(
        "--model_only",
        default=False,
        action='store_true',
        help="load model weights in checkpoint only, no optimizer, scheduler and epoch",
    )
    parser.add_argument(
        "--start_iter",
        default=0,
        type=int,
        help="fix starting iteration if loading model_only",
    )
    parser.add_argument(
        "--gpu",
        default=-1,
        type=int,
        help="specify which gpu to use",
    )
    parser.add_argument(
        "--save_images_locally",
        default=False,
        action='store_true',
        help="flag to save images locally",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    run_id = random.randint(1, 100000)
    logdir = os.path.join("runs", os.path.basename(args.config)[:-4], "cur")
    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Let the games begin")

    if args.save_images_locally:
        save_dir = os.path.join(logdir, 'image_logs')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    else:
        save_dir = None
        print('No save directory specified to log images locally.')

    train(cfg, writer, logger, args.start_iter, args.model_only, gpu = args.gpu, save_dir = save_dir)
