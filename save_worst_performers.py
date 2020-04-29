import os
import glob
import pandas as pd
import numpy as np
import cv2
import argparse


def save_worst_performing_images(args):
    """ Reads imagewise metric from csv file output (validate.py)
    Saves bottom percentile of images with metric score on image
    """

    miou_df = pd.read_csv(args.csv_path)
    miou_df = miou_df.sort_values(by=[args.target_metric])

    n_images_to_copy = int(args.perc * len(miou_df))
    tar_images = miou_df[['filename', args.target_metric]][:n_images_to_copy].round(3).values.tolist()

    all_copy_paths = [
        f for f in glob.glob(
            os.path.join(args.dataset_path, "**/*.*"), # finds paths with file extensions only
            recursive=True
        )
    ]

    copy_paths = [] # stores tuples, with imagewise data required for logging
    for copy_path in all_copy_paths:
        for img_path, img_score in tar_images:
            copy_path_basename = os.path.splitext(copy_path)[0]
            img_path_basename = os.path.splitext(img_path)[0]
            if copy_path_basename.endswith(img_path_basename):  # ignore file type extension when matching names
                copy_paths.append((copy_path, img_path, img_score))

    copy_paths.sort(key = lambda x : x[2]) # sort in order of increasing score, match order for raw metric outputs
    assert len(copy_paths) == len(tar_images), 'Number of files to copy %s not equal to number of matching files in copy_path %s, Sample copy target string %s' %(len(tar_images), len(copy_paths), tar_images[0])
    if args.raw_metric is not None:
        raw_metrics = miou_df[args.raw_metric][:n_images_to_copy].values.tolist()
        copy_paths = [ data + (raw_metric,) for data, raw_metric in zip(copy_paths, raw_metrics) ]

    output_dir = os.path.join( os.path.dirname(args.csv_path), 'worst_%.1f_%s' %(args.perc, args.target_metric))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    i = 0
    for data in copy_paths:
        copy_path = data[0]
        img_score = data[2]
        raw_outputs = ""
        if args.raw_metric is not None:
            raw_outputs = data[3]
        
        paste_path = os.path.join(output_dir, "%05d.jpg" %i)
        image = cv2.imread(copy_path)
        image = cv2.putText(image, "%s: %s, %s" %(args.target_metric, img_score, raw_outputs), 
            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2
        )
        cv2.imwrite(paste_path, image)
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
        '--raw_metric',
        type=str,
        default=None,
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