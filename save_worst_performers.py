import os
import glob
import pandas as pd
import numpy as np
from shutil import copy
import argparse


def save_worst_performing_images(args):

    miou_df = pd.read_csv(args.csv_path)
    miou_df = miou_df.sort_values(by=[args.target_metric])

    n_images_to_copy = int(args.perc * len(miou_df))
    tar_images = miou_df['filename'][:n_images_to_copy].values.tolist()

    copy_paths = [
        f for f in glob.glob(
            os.path.join(args.dataset_path, "**/*.*"), # finds paths with file extensions only
            recursive=True
        ) if True in 
        [os.path.splitext(f)[0].endswith( os.path.splitext(img_path)[0] ) # ignore file type
        for img_path in tar_images]
    ]
    output_dir = os.path.join( os.path.dirname(args.csv_path), 'worst_%.1f_%s' %(args.perc, args.target_metric))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    assert len(copy_paths) == len(tar_images), 'Number of files to copy %s not equal to number of matching files in copy_path %s, Sample copy target string %s' %(len(tar_images), len(copy_paths), tar_images[0])

    i = 0
    for copy_path in copy_paths:
        paste_path = os.path.join(output_dir, "%05d.jpg" %i)
        copy(copy_path, paste_path)
        i += 1
        
        print("%s -> %s" %(copy_path, paste_path))


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
        '--target_metric',
        required=True,
        type=str,
        help='Csv heading to take calculate bottom percentile'
    )
    parser.add_argument(
        '--perc',
        required=True,
        type=float,
        help='Percentage of images to output to folder'
    )
    args = parser.parse_args()

    save_worst_performing_images(args)