import yaml
import torch
import argparse
import timeit
import time
import os
import numpy as np
import imageio

from torch.utils import data
from torchstat import stat
from pytorch_bn_fusion.bn_fusion import fuse_bn_recursively

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScoreSeg
from ptsemseg.utils import convert_state_dict
from ptsemseg.augmentations import get_composed_augmentations, get_composed_augmentations_softmax

torch.backends.cudnn.benchmark = True

def reset_batchnorm(m):
    if isinstance(m, torch.nn.BatchNorm2d):
      m.reset_running_stats()
      m.momentum = None

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

    if "version" in cfg["data"]:
        version = cfg["data"]["version"]
    else:
        version = "cityscapes"

    loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg["data"]["val_split"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        augmentations=data_aug,
        version=version,
    )

    n_classes = loader.n_classes

    valloader = data.DataLoader(loader, batch_size=1, num_workers=1)
    running_metrics = runningScoreSeg(n_classes)

    # Setup Model

    model = get_model(cfg["model"], n_classes).to(device)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)
    
    if args.bn_fusion:
      model = fuse_bn_recursively(model)
      print(model)
    
    if args.update_bn:
      print("Reset BatchNorm and recalculate mean/var")
      model.apply(reset_batchnorm)
      model.train()
    else:
      model.eval()
    model.to(device)
    total_time = 0
    
    total_params = sum(p.numel() for p in model.parameters())
    print('Parameters: ', total_params )
    
    #stat(model, (3, 1024, 2048))
    torch.backends.cudnn.benchmark=True

    with open(args.output_csv_path, 'a') as output_csv:

        output_csv.write('filename,miou,fps\n')

        for i, (images, labels, fname) in enumerate(valloader):
            start_time = timeit.default_timer()

            images = images.to(device)
            
            if i == 0:
                with torch.no_grad():
                    outputs = model(images)        
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                outputs = model(images)

            torch.cuda.synchronize()
            elapsed_time = time.perf_counter() - start_time
            
            if args.save_image:
                pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
                
                decoded = loader.decode_segmap_id(pred)
                dir = "./out_predID/"
                if not os.path.exists(dir):
                    os.mkdir(dir)
                    imageio.imwrite(dir+fname[0], decoded)

                decoded = loader.decode_segmap(pred)
                img_input = np.squeeze(images.cpu().numpy(),axis=0)
                img_input = img_input.transpose(1, 2, 0)
                blend = img_input * 0.2 + decoded * 0.8
                fname_new = fname[0]
                fname_new = fname_new[:-4]
                fname_new += '.jpg'
                dir = "./out_rgb/"
                
                if not os.path.exists(dir):
                    os.mkdir(dir)
                if not os.path.exists(os.path.join(dir, fname_new.split(os.sep)[0])):
                    os.mkdir( os.path.join(dir, fname_new.split(os.sep)[0]) )
                imageio.imwrite(dir+fname_new, blend)

            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels.numpy()
            s = np.sum(gt==pred) / (cfg["data"]["img_rows"] * cfg["data"]["img_cols"] - np.sum(gt == 250)) # consider ignore label == 250

            if args.measure_time:
                total_time += elapsed_time
                print(
                    "Inference time \
                    (iter {0:5d}): {1:4f}, {2:3.5f} fps".format(
                        i + 1, s,1 / elapsed_time
                    )
                )
            
            output_csv.write( '%s, %.4f, %.4f\n' %(fname[0], s, 1 / elapsed_time) )

            running_metrics.update(gt, pred)

    score, class_iou = running_metrics.get_scores()
    print("Total Frame Rate = %.2f fps" %(500/total_time ))

    if args.update_bn:
      model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
      state2 = {"model_state": model.state_dict()}
      torch.save(state2, 'hardnet_cityscapes_mod.pth')

    with open(args.miou_logs_path, 'a') as main_output_csv:
        main_output_csv.write( '%s\n' %args.output_csv_path )

        for k, v in score.items():
            print(k, v)
            main_output_csv.write( '%s,%s\n' %(k,v) )

        for i in range(n_classes):
            print(i, class_iou[i])
            main_output_csv.write( '%s,%s\n' %(i, class_iou[i]) )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/hardnet.yml",
        help="Config file to be used",
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="hardnet_cityscapes_best_model.pkl",
        help="Path to the saved model",
    )
    
    parser.add_argument(
        "--measure_time",
        dest="measure_time",
        action="store_true",
        help="Enable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.add_argument(
        "--no-measure_time",
        dest="measure_time",
        action="store_false",
        help="Disable evaluation with time (fps) measurement",
    )
    parser.set_defaults(measure_time=True)

    parser.add_argument(
        "--save_image",
        dest="save_image",
        action="store_true",
        help="Enable saving inference result image into out_img/ |\
                              False by default",
    )
    parser.set_defaults(save_image=False)
    
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

    if not args.config.endswith('yml'):
        cfgs = os.listdir(args.config)
    else:
        cfgs = [args.config]

    args.miou_logs_path = os.path.join(
        os.path.dirname(args.model_path),
        'overall_miou_logs.csv'
    )

    for cfg_filename in cfgs:
        
        cfg_filename = os.path.join(args.config, cfg_filename)

        with open(cfg_filename) as fp:
            cfg = yaml.load(fp)
        
        args.output_csv_path = os.path.join(
            os.path.dirname(args.model_path), 
            'perimage_logs_%s.csv' %os.path.basename(cfg_filename)[:-4]
        )
        validate(cfg, args)