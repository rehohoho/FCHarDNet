import yaml
import glob
import os
import numpy as np
from PIL import Image
import argparse

import torch
from ptsemseg.models.hardnet import hardnet


def get_tensornames_to_txtfile(model_path, output_path):
    model = torch.load(model_path)
    output = []
    for i in model['model_state']:
        output.append( i+':'+ str(np.shape(model['model_state'][i])) )
    with open(output_path, 'w') as f:
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


def process_img(img, resize = False):

    img = np.array(img, dtype=np.float64)
    img = img[:, :, ::-1]  # RGB -> BGR
    
    mean = [0.406, 0.456, 0.485]
    mean = [item * 255 for item in mean]
    std = [0.225, 0.224, 0.229]
    std = [item * 255 for item in std]
    img = (img - mean)/std

    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()

    return(img)


def inference_on_folder(folder_path, output_path, model_path, resize):

    # load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = hardnet(n_classes=19).to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    weights = torch.load(model_path)
    model.load_state_dict(weights['model_state'])
    print('model is loaded')

    if not os.path.exists(output_path):
        os.mkdir(output_path)
        print('output directory not found, created one %s' %output_path)

    filenames = [file for file in 
        glob.glob( os.path.join(folder_path, '**/*.png'), recursive=True )
    ]

    for filename in filenames:
        
        dirname = os.path.join( output_path, filename.split('/')[-2] )
        if not os.path.exists(dirname):
            print('Creating directory at %s' %dirname)
            os.mkdir(dirname)

        img = Image.open(filename)
        
        if resize:
            width, height = img.size                # resizing such that max = max_input_size
            resize_ratio = 513 / max(width, height)
            target_size = (int(resize_ratio * width), int(resize_ratio * height))
            img = img.convert('RGB').resize(target_size, Image.ANTIALIAS)

        img_tensor = process_img(img, resize = resize)
        outputs = model(img_tensor.unsqueeze(0))
        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
        seg_im = CITYSCAPES_COLORMAP[pred]

        seg_im = np.hstack((img, seg_im))
        
        basename = os.path.join( filename.split('/')[-2], os.path.basename(filename) )
        Image.fromarray(seg_im).save( os.path.join(output_path, basename) )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-f', '--folder_path',
        required=True,
        help= "path to folder with images"
    )
    parser.add_argument(
        '-o', '--output_path',
        required=True,
        help= "path to folder to output"
    )
    parser.add_argument(
        '-m', '--model_path',
        required=True,
        help= "path to fchardnet model"
    )
    parser.add_argument(
        '-r', '--resize',
        default=False,
        action='store_true',
        help= "option to resize"
    )

    args = parser.parse_args()
    inference_on_folder(args.folder_path, args.output_path, args.model_path, args.resize)