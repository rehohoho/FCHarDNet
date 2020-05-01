"""
Seperate inference script for inference on test images without labels
No metrics will be logged since ground truths are not available.
"""

import yaml
import glob
import os
import numpy as np
from scipy.special import softmax
import cv2
import re
import argparse

import torch
from ptsemseg.models.hardnet import hardnet
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


HEAD_OUTPUT_MAP = {
    "environment": {
        0: "bus stop",
        1: "junction",
        2: "road break",
        3: "straight pavement"
    }
}

CITYSCAPES_COLORMAP = np.array( [
    [128, 64,128],  #road
    [244, 35,232],  #sidewalk
    [ 70, 70, 70],  #building
    [102,102,156],  #wall
    [190,153,153],  #fence
    [153,153,153],  #pole
    [250,170, 30],  #traffic light
    [220,220,  0],  #traffic sign
    [107,142, 35],  #vegetation
    [152,251,152],  #terrain
    [ 70,130,180],  #sky
    [220, 20, 60],
    [255,  0,  0],
    [  0,  0,142],
    [  0,  0, 70],
    [  0, 60,100],
    [  0, 80,100],
    [  0,  0,230],
    [119, 11, 32]
    
    ], dtype = np.uint8)


def get_tensornames_to_txtfile(model_directory, output_folder):
    model = torch.load(model_directory)
    output = []
    for i in model['model_state']:
        output.append( i+':'+ str(np.shape(model['model_state'][i])) )
    with open(output_folder, 'w') as f:
        f.write( '\n'.join(output) )


def _sort_dir_alphanum(im_dir):
    # sort directory based on the alphanumeric value of the direcory
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(im_dir, key=alphanum_key)


def _get_image_tensor(img, cfg):
    """ Resize image according to model config, 
    and standardise to get image tensor for model input
    
    Args
        img             3D np.ndarray
        cfg             model config dictionary from yaml
    Returns
        img             3D np.ndarray, resized rgb image
        img_tensor      3D tensor, resized, standardised image tensor
    """

    img = cv2.resize(cv2.imread(img), (cfg["data"]["img_rows"], cfg["data"]["img_cols"]))
    img_tensor = np.array(img, dtype=np.float64)

    mean = [0.406, 0.456, 0.485]
    mean = [item * 255 for item in mean]
    std = [0.225, 0.224, 0.229]
    std = [item * 255 for item in std]
    img_tensor = (img_tensor - mean)/std

    img_tensor = img_tensor.transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img_tensor).float()

    return img, img_tensor


def _add_message_to_image(image, hist_data_dict):
    """ expects hist_data_dict of the following structure
    hist_data_dict = {
        head_name: {
            raw: list(np.ndarray)
            pred: list(string)    }}
    """

    msg = []
    for name, data in hist_data_dict.items():
        msg.append("%s: %s" %(name, data["pred"][-1]))
        msg.append("%s: %s" %(
            name, ",".join([str(i) for i in data["raw"][-1]])
        ))

    pos = 50
    for line in msg:
        image = cv2.putText(image, line,
                    (50, pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        pos += 50
    
    return image


def _plot_raw_class_outputs(image, hist_data_dict):
    
    height, width, _ = image.shape
    tar_graph_size = min(height/2, width/2)

    fig = Figure(figsize=[width/100, height/100])
    canvas = FigureCanvas(fig)
    axes = fig.add_axes( [0,0,1,1] )
    axes.imshow(image)

    graph_idx = 0
    for name, hist_data in hist_data_dict.items():
        data = np.array(hist_data["raw"])
        n_points, n_components = data.shape
        components = np.split(data, n_components, axis=1)
        
        x_axis = np.arange(n_points) * tar_graph_size/10 + 50 + graph_idx*tar_graph_size   
        for i in range(len(components)):
            y_axis = height - components[i].squeeze()*tar_graph_size - 1
            axes.plot(x_axis, y_axis, label=HEAD_OUTPUT_MAP[name][i], linewidth=3.0)
        
        graph_idx += 1
    
    axes.legend()
    canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    image = image[:,:,::-1]
    
    return image


def inference_on_folder(image_folder, output_folder, model_directory, cfg,
                        vis = True, add_orig = True):

    # load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param_dict = cfg["model"]
    param_dict.pop("arch")
    model = hardnet(n_classes = 19, **param_dict)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    weights = torch.load(model_directory, map_location="cuda:0")
    model.load_state_dict(weights['model_state'])
    print('model is loaded')

    # load the images
    filenames = [file for file in 
        glob.glob( os.path.join(image_folder, '**/*.png'), recursive=True )
    ]
    filenames = _sort_dir_alphanum(filenames)
    print("Number of images found: %s" %len(filenames))

    head_outputs = {}
    for filename in filenames:

        save_path = filename.replace(image_folder, output_folder)
        
        if not os.path.exists(save_path):
            
            print('%s -> %s' %(filename, save_path))
            
            resized_img, img_tensor = _get_image_tensor(filename, cfg)
            output_dict = model(img_tensor.unsqueeze(0))
            
            for name, output in output_dict.items():
                if name in HEAD_OUTPUT_MAP.keys():
                    raw_scores = softmax(np.squeeze(
                        output.data.cpu().numpy()
                    )).round(3)

                    if name not in head_outputs.keys():
                        head_outputs[name] = {"raw": [], "pred": []}
                    head_outputs[name]["raw"].append( raw_scores )
                    head_outputs[name]["pred"].append( HEAD_OUTPUT_MAP[name][np.argmax(raw_scores)] )

                    if len(head_outputs[name]["raw"]) > 10:
                        del head_outputs[name]["raw"][0]
                        del head_outputs[name]["pred"][0]

            seg_im = np.squeeze(output_dict["seg"].data.max(1)[1].cpu().numpy(), axis=0).astype(np.uint8)

            if vis:
                seg_im = CITYSCAPES_COLORMAP[seg_im]

            if add_orig:
                seg_im = np.hstack((resized_img[:,:,::-1], seg_im)) # _plot_raw_class_outputs requires seg_im to be rgb
            
            seg_im = _add_message_to_image(seg_im, head_outputs)
            seg_im = _plot_raw_class_outputs(seg_im, head_outputs)

            dirname = os.path.dirname(save_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            
            cv2.imwrite(save_path, seg_im)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--image_folder',
        required=True,
        help="path to folder with images"
    )
    parser.add_argument(
        '--output_folder',
        required=True,
        help="path to folder to output"
    )
    parser.add_argument(
        '--model_directory',
        required=True,
        help="path to fchardnet model"
    )
    parser.add_argument(
        '--model_config',
        required=True,
        help='path to model config file (.yml)'
    )
    parser.add_argument(
        '--vis_mask',
        default=False,
        action='store_true',
        help= "flag to turn mask into visualisation"
    )
    parser.add_argument(
        '--add_orig',
        default=False,
        action='store_true',
        help= "flag to attach segmentation image with original image"
    )

    args = parser.parse_args()

    with open(args.model_config) as fp:
        cfg = yaml.load(fp)

    inference_on_folder(args.image_folder, args.output_folder, args.model_directory, cfg,
                        vis = args.vis_mask, add_orig = args.add_orig)