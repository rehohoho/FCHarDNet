python inference.py \
    --model_directory='D:/models/scooter_resized_halflabelled_180.pkl' \
    --image_folder='D:/data/detector/detector_images' \
    --output_folder='D:/data/detector/detector_vis' \
    --tar_size=513 \
    --vis_mask \
    --add_orig

# MAIN ARGUMENTS
    # --image_folder        # path to folder with images
    # --output_folder       # path to folder for segmented images
    # --model_directory     # path to the directory with tar.gz model checkpoint"

# VISUALISATION ARGUMENTS
    # --vis_mask            # flag to turn mask into visualisation
    # --add_orig            # flag to attach segmentation image with original