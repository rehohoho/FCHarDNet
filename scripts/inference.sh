python ../inference.py \
    --model_directory="<path-to-model-pkl-file>" \
    --model_config="<path-to-model-yml-file>" \
    --image_folder="<path-to-image-directory>" \
    --output_folder="<path-to-output-directory>" \
    --vis_mask \
    --add_orig

# MAIN ARGUMENTS
    # --image_folder        # path to folder with images
    # --output_folder       # path to folder for segmented images
    # --model_directory     # path to the directory with tar.gz model checkpoint"
    # --model_config        # path to model config file (.yml)

# VISUALISATION ARGUMENTS
    # --vis_mask            # flag to turn mask into visualisation
    # --add_orig            # flag to attach segmentation image with original

# MAPPING OF CLASSIFIER HEADS HAS TO BE SPECIFIED IN THE SCRIPT