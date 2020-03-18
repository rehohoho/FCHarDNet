import os
import pandas as pd
import numpy as np
from shutil import copy
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_path',
        required=True,
        type=str,
        help='Path to csv file with miou outputs from validate.py'
    )
    parser.add_argument(
        '--csv_path',
        required=True,
        type=str,
        help='Path to csv file with miou outputs from validate.py'
    )
    parser.add_argument(
        '--perc',
        required=True,
        type=float,
        help='Percentage of images to output to folder'
    )
    args = parser.parse_args()

    miou_df = pd.read_csv(args.csv_path)
    miou_df = miou_df.sort_values(by=['miou'])
    
    n_images_to_copy = int(args.perc * len(miou_df))
    tar_images = miou_df['filename'][:n_images_to_copy].values.tolist()

    output_dir = os.path.join( os.path.dirname(args.csv_path), 'worst_%.1f_miou' %args.perc)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print('Created directory at %s' %output_dir)

    for img_path in tar_images:

        path_copy_from = os.path.join(args.dataset_path, img_path)
        path_copy_to = os.path.join(output_dir, img_path)
        path_copy_to_dir = os.path.dirname(path_copy_to)
        
        if not os.path.exists(path_copy_to_dir):
            os.mkdir(path_copy_to_dir)
        
        copy(path_copy_from, path_copy_to)
