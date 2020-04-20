import yaml
import glob
import os
import numpy as np
from PIL import Image
import argparse

import torch
from ptsemseg.models.hardnet import hardnet


def get_tensornames_to_txtfile(model_directory, output_folder):
    model = torch.load(model_directory)
    output = []
    for i in model['model_state']:
        output.append( i+':'+ str(np.shape(model['model_state'][i])) )
    with open(output_folder, 'w') as f:
        f.write( '\n'.join(output) )


# LOAD AND INFER
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


def process_img(img, tar_size = None):
    """ Resize image, and standardise to get image tensor for model input
    
    Args
        img             3D np.ndarray
        tar_size        int, model input dims
    Returns
        img             3D np.ndarray, resized rgb image
        img_tensor      3D tensor, resized, standardised image tensor
    """

    if tar_size is not None:
        width, height = img.size                # resizing such that max = max_input_size
        resize_ratio = tar_size / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        img = img.convert('RGB').resize(target_size, Image.ANTIALIAS)
    
    img_tensor = np.array(img, dtype=np.float64)
    img_tensor = img_tensor[:, :, ::-1]  # RGB -> BGR
    
    mean = [0.406, 0.456, 0.485]
    mean = [item * 255 for item in mean]
    std = [0.225, 0.224, 0.229]
    std = [item * 255 for item in std]
    img_tensor = (img_tensor - mean)/std

    img_tensor = img_tensor.transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img_tensor).float()

    return img, img_tensor


def inference_on_folder(image_folder, output_folder, model_directory, 
                        tar_size = None, vis = True, add_orig = True):

    # load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = hardnet(n_classes=19).to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    weights = torch.load(model_directory, map_location="cuda:0")
    model.load_state_dict(weights['model_state'])
    print('model is loaded')

    filenames = [file for file in 
        glob.glob( os.path.join(image_folder, '**/*.png'), recursive=True )
    ]

    print(len(filenames))

    for filename in filenames:

        save_path = filename.replace(image_folder, output_folder)
        
        if not os.path.exists(save_path):
            
            print('%s -> %s' %(filename, save_path))

            img = Image.open(filename)
            
            resized_img, img_tensor = process_img(img, tar_size = tar_size)
            outputs = model(img_tensor.unsqueeze(0))
            seg_im = np.squeeze(outputs["seg"].data.max(1)[1].cpu().numpy(), axis=0).astype(np.uint8)
            
            if vis:
                seg_im = CITYSCAPES_COLORMAP[seg_im]
            
            if add_orig:
                seg_im = np.hstack((resized_img, seg_im))
            
            dirname = os.path.dirname(save_path)

            if not os.path.exists(dirname):
                os.makedirs(dirname)
            
            Image.fromarray(seg_im).save(save_path)


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
        '--tar_size',
        default=None,
        type=int,
        help="size of input mask"
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
    inference_on_folder(args.image_folder, args.output_folder, args.model_directory, 
                        tar_size = args.tar_size, vis = args.vis_mask, add_orig = args.add_orig)