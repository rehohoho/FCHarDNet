import yaml
import torch
import argparse
import time
import os
import numpy as np
from scipy.special import softmax
from PIL import Image

from torch.utils import data
from torchstat import stat
from pytorch_bn_fusion.bn_fusion import fuse_bn_recursively

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScoreSeg, runningScoreClassifier
from ptsemseg.utils import convert_state_dict, get_cityscapes_image_from_tensor
from ptsemseg.augmentations import get_composed_augmentations, get_composed_augmentations_softmax

torch.backends.cudnn.benchmark = True


def reset_batchnorm(m):
    if isinstance(m, torch.nn.BatchNorm2d):
      m.reset_running_stats()
      m.momentum = None


def save_image(images, output_dict, fname, output_path, loader):

    pred = np.squeeze(output_dict["seg"].data.max(1)[1].cpu().numpy(), axis=0)
    img_input = np.squeeze(images.cpu().numpy(),axis=0).transpose(1, 2, 0) # mask overlay image
    
    mask = loader.decode_segmap(loader.decode_segmap_id(pred)) # visualisation of mask
    mask = get_cityscapes_image_from_tensor(mask, mask=True, get_image_obj = False)
    img = get_cityscapes_image_from_tensor(img_input, get_image_obj = False)
    img = np.hstack((img, mask))
    
    save_path = os.path.join(output_path+"_images", "%s.jpg" %fname[0][:-4])
    save_path_dir = os.path.dirname(save_path)
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
    Image.fromarray(img).save(save_path)


def create_overall_logs_header(running_metrics_val):
    header = ["filename", "fps"]
    for name in running_metrics_val.keys():
        header.append(name)
        if name != "seg":
            header.append("%s_output" %name)
            header.append("%s_confusion" %name)
    
    return "%s\n" %(",".join(header))


def validate(cfg, args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Augmentations
    data_aug = None
    if "validation" in cfg:
        augmentations = cfg["validation"].get("augmentations", None)
        if cfg["data"]["dataset"] == "softmax_cityscapes_convention":
            data_aug = get_composed_augmentations_softmax(augmentations)
        else:
            data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    loader = data_loader(
        data_path,
        config = cfg["data"],
        is_transform=True,
        split=cfg["data"][args.dataset_split],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        augmentations=data_aug,
    )
    n_classes = loader.n_classes
    valloader = data.DataLoader(loader, batch_size=1, num_workers=1)
    
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
    state = torch.load(args.model_path, map_location="cuda:0")["model_state"]
    state = convert_state_dict(state) # converts from dataParallel module to normal module
    model.load_state_dict(state, strict=False)
    
    if args.bn_fusion:
      model = fuse_bn_recursively(model)
    
    if args.update_bn:
      print("Reset BatchNorm and recalculate mean/var")
      model.apply(reset_batchnorm)
      model.train()
    else:
      model.eval() # set batchnorm and dropouts to work in eval mode
    model.to(device)
    total_time = 0
    
    total_params = sum(p.numel() for p in model.parameters())
    print('Parameters: ', total_params )
    
    #stat(model, (3, 1024, 2048))
    torch.backends.cudnn.benchmark=True

    with open(args.output_csv_path, 'a') as output_csv:

        output_csv.write(create_overall_logs_header(running_metrics_val))

        for i, (images, label_dict, fname) in enumerate(valloader):
            images = images.to(device)
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            with torch.no_grad(): # deactivates autograd engine, less mem usage
                output_dict = model(images)
            torch.cuda.synchronize()
            elapsed_time = time.perf_counter() - start_time
            
            if args.save_image:
                save_image(images, output_dict, fname, args.output_path, loader=loader)
            
            image_score = []
            
            for name, metrics in running_metrics_val.items(): # update running metrics and record imagewise metrics
                gt_array = label_dict[name].data.cpu().numpy()
                if name+'_loss' in cfg['training'] and cfg['training'][name+'_loss']['name'] == 'l1': # for binary classification
                    pred_array = output_dict[name].data.cpu().numpy()
                    pred_array = np.sign(pred_array)
                    pred_array[pred_array == -1] = 0
                    gt_array[gt_array == -1] = 0
                else:
                    pred_array = output_dict[name].data.max(1)[1].cpu().numpy()

                if name == "seg" or name == "softmax":
                    image_score.append( "%.3f" %metrics.get_image_score(gt_array, pred_array) )
                else:
                    imagewise_score = softmax(np.squeeze(
                        output_dict[name].data.cpu().numpy()
                    )).round(3)
                    image_score.append( "%.3f" %(imagewise_score[gt_array[0]]) )
                    image_score.append( str(imagewise_score) ) # append raw probability results for non-segmentation task
                    image_score.append( "pred %s label %s" %(np.argmax(imagewise_score), gt_array[0]))
                
                metrics.update(gt_array, pred_array)

            output_csv.write( '%s, %.4f, %s\n' %(fname[0], 1 / elapsed_time, ",".join(image_score)) ) # record imagewise metrics

            if args.measure_time:
                total_time += elapsed_time
                print(
                    "Iter {0:5d}: {1:3.5f} fps {2}".format(
                        i + 1, 1 / elapsed_time, " ".join(image_score)
                    )
                )

    print("Total Frame Rate = %.2f fps" %(i/total_time ))

    if args.update_bn:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        state2 = {"model_state": model.state_dict()}
        torch.save(state2, 'hardnet_cityscapes_mod.pth')

    with open(args.miou_logs_path, 'a') as main_output_csv: # record overall metrics
        main_output_csv.write( '%s\n' %args.output_csv_path )

        for name, metrics in running_metrics_val.items():
            overall, classwise = metrics.get_scores()
            
            for k, v in overall.items():
                print("{}_{}: {}".format(name, k, v))
                main_output_csv.write("%s,%s,%s\n" %(name, k, v))

            for metric_name, metric in classwise.items():
                for k, v in metric.items():
                    print("{}_{}_{}: {}".format(name, metric_name, k, v))
                    main_output_csv.write( "%s,%s,%s,%s\n" %(name, metric_name, k, v))
            
            confusion_matrix = np.round(metrics.confusion_matrix, 3)
            print("confusion matrix:\n%s" %confusion_matrix)
            main_output_csv.write("%s\n" %(
                "\n".join(str(i) for i in confusion_matrix)
            ))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Hyperparams")
    
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/hardnet.yml",
        help="Config file corresponding to model. Required to build model.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="weights/hardnet70_cityscapes_model.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--aug_configs",
        type=str,
        default=None,
        help="Directory of configs or config containing augmentation strategy to apply."
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="val_split",
        help="train_split or val_split"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to output logs and images"
    )
    parser.add_argument(
        "--measure_time",
        action="store_true",
        default=False,
        help="Enable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.add_argument(
        "--save_image",
        action="store_true",
        default=False,
        help="Enable saving inference result image into out_img/ |\
                              False by default",
    )

    parser.add_argument(
        "--update_bn",
        dest="update_bn",
        action="store_true",
        help="Reset and update BatchNorm running mean/var with entire dataset |\
              False by default",
    )
    parser.set_defaults(update_bn=False)
    
    parser.add_argument(
        "--no-bn_fusion",
        dest="bn_fusion",
        action="store_false",
        help="Disable performing batch norm fusion with convolutional layers |\
              bn_fusion is enabled by default",
    )
    parser.set_defaults(bn_fusion=True)   

    args = parser.parse_args()

    # Setup model and augmentation configs
    with open(args.model_config) as fp:
        model_cfg = yaml.load(fp)
    
    if not args.aug_configs.endswith('yml'):
        aug_cfgs = os.listdir(args.aug_configs)
    else:
        aug_cfgs = [args.aug_configs]
    
    # Setup main logging paths
    if args.output_path is None:
        base_output_path = os.path.join(
            os.path.dirname(args.model_path),
            os.path.splitext(os.path.basename(args.model_path))[0]
        )
    else:
        base_output_path = args.output_path
    if not os.path.exists(base_output_path):
        os.makedirs(base_output_path)
    args.miou_logs_path = os.path.join(base_output_path, 'overall_miou_logs.csv')

    for aug_cfg_filename in aug_cfgs:
        
        aug_cfg_filename = os.path.join(args.aug_configs, aug_cfg_filename)
        with open(aug_cfg_filename) as fp:
            aug_cfg = yaml.load(fp)
        
        cfg = dict(model_cfg)
        if "validation" in aug_cfg:
            cfg.update({"validation": aug_cfg["validation"]})

        args.output_csv_path = os.path.join(
            base_output_path,
            'perimage_logs_%s.csv' %os.path.basename(aug_cfg_filename)[:-4]
        )
        args.output_path = os.path.join(
            base_output_path,
            os.path.basename(aug_cfg_filename)[:-4]
        )
        validate(cfg, args)