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
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations, get_composed_augmentations_softmax
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer

from tensorboardX import SummaryWriter


def write_images_to_board(loader, i_val, image, gt, pred, step, name):
    
    writer_label = loader.decode_segmap(gt) #takes HW nd.array, outputs HWC
    writer_label = writer_label.transpose(2, 0, 1) #change to CHW
    writer_label = torch.Tensor(writer_label).type('torch.cuda.FloatTensor')
    
    writer_pred = loader.decode_segmap(pred)
    writer_pred = writer_pred.transpose(2, 0, 1)
    writer_pred = torch.Tensor(writer_pred).type('torch.cuda.FloatTensor')

    writer.add_image('%s_%s_Image' %(name, i_val), image, step)
    writer.add_image('%s_%s_Label' %(name, i_val), writer_label, step)
    writer.add_image('%s_%s_Pred' %(name, i_val), writer_pred, step)


def get_image_from_tensor(image, mask = False):
    
    if mask:
        image *= 255
    else:
        std = np.array([57.375, 57.12 , 58.395])
        mean = np.array([103.53 , 116.28 , 123.675])
        image = (image * std + mean)
    
    image = Image.fromarray(image.astype(np.uint8))
    return image


def write_images_to_dir(loader, image, gt, pred, step, save_dir, name):
    
    writer_label = [loader.decode_segmap(i)
                            for i in gt]
    
    writer_pred = [ loader.decode_segmap(i)
                            for i in pred]
    
    save_path = os.path.join(save_dir, str(step))
    
    for i in range(len(image)):
        get_image_from_tensor(image[i]).save('%s_%s_%d_Image.png' %(save_path, name, i))
        get_image_from_tensor(writer_label[i], mask=True).save('%s_%s_%d_Label.png' %(save_path, name, i))
        get_image_from_tensor(writer_pred[i], mask=True).save('%s_%s_%d_Pred.png' %(save_path, name, i))


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)

def train(cfg, writer, logger, start_iter=0, model_only=False, gpu=-1, save_dir=None):

    # Setup seeds and config
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))
    
    use_softmax_labels = cfg["data"]["dataset"] == "softmax_cityscapes_convention"

    # Setup device
    if gpu == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:%d" %gpu if torch.cuda.is_available() else "cpu")

    # Setup Augmentations
    augmentations = cfg["training"].get("augmentations", None)
    if use_softmax_labels:
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
        is_transform=True,
        split=cfg["data"]["train_split"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        augmentations=data_aug,
        version=version,
    )

    v_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg["data"]["val_split"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        version=version,
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
    running_metrics_val = runningScore(n_classes)

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

    if 'weight' in cfg['training']['loss']:
        cfg['training']['loss']['weight'] = torch.Tensor(cfg['training']['loss']['weight']).to(device)
    loss_fn = get_loss_function(cfg)
    print("Using loss {}".format(loss_fn))

    if cfg["training"]["resume"] is not None:
        if os.path.isfile(cfg["training"]["resume"]):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
            )
            checkpoint = torch.load(cfg["training"]["resume"])
            model.load_state_dict(checkpoint["model_state"])
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
    max_n_images = 10
    max_n_batches = 1

    while i <= cfg["training"]["train_iters"] and flag:
        i_train = 0
        for (images, labels, _) in trainloader:
            i += 1
            start_ts = time.time()
            scheduler.step()
            model.train()
            images = images.to(device)  #n,3,513,513
            labels = labels.to(device)  #n,19,513,513

            optimizer.zero_grad()
            outputs = model(images)     #n,19,513,513

            # log images, labels, outputs
            if save_dir is not None and i_train <= max_n_images:
                pred_array = outputs.data.max(1)[1].cpu().numpy()     #1,513,513
                if use_softmax_labels:
                    gt_array = labels.data.max(1)[1].cpu().numpy()    #1,513,513
                else:
                    gt_array = labels.data.cpu().numpy()
                write_images_to_board(t_loader, i_train, images[0], gt_array[0], pred_array[0], i, 'train')

                if i_train <= max_n_batches:
                    image_array = images.data.cpu().numpy().transpose(0, 2, 3, 1)    #513,513,3
                    write_images_to_dir(t_loader, image_array, gt_array, pred_array, i, save_dir, name='train')

            if use_softmax_labels: # has to be done outside loss function where image is not passed in
                loss = loss_fn(input=outputs, target=labels, weight=t_loader.extract_ignore_mask(images))
            else:
                loss = loss_fn(input=outputs, target=labels)

            loss.backward()
            optimizer.step()
            c_lr = scheduler.get_lr()

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
                    for i_val, (images_val, labels_val, _) in tqdm(enumerate(valloader)):
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)

                        outputs = model(images_val)
                        val_loss = loss_fn(input=outputs, target=labels_val)

                        pred_array = outputs.data.max(1)[1].cpu().numpy()
                        gt_array = labels_val.data.cpu().numpy()

                        running_metrics_val.update(gt_array, pred_array)
                        val_loss_meter.update(val_loss.item())

                        # log validation images
                        if save_dir is not None and i_val <= max_n_images:
                            write_images_to_board(v_loader, i_val, images_val[0], gt_array[0], pred_array[0], i, 'validation')
                            if i_val <= max_n_batches:
                                images_val = images_val.cpu().numpy().transpose(0, 2, 3, 1)
                                write_images_to_dir(v_loader, images_val, gt_array, pred_array, i, save_dir, name='validation')

                writer.add_scalar("loss/val_loss", val_loss_meter.avg, i + 1)

                logger.info("Iter %d Val Loss: %.4f" % (i + 1, val_loss_meter.avg))

                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k, v)
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/{}".format(k), v, i + 1)

                for k, v in class_iou.items():
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/cls_{}".format(k), v, i + 1)

                val_loss_meter.reset()
                running_metrics_val.reset()
                
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

                if score["Mean IoU : \t"] >= best_iou:
                    best_iou = score["Mean IoU : \t"]
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
