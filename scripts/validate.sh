python ../validate.py \
    --model_config="<path-to-config-yml-file>" \
    --model_path="<path-to-model-pkl-file>" \
    --aug_configs=../configs/aug_val \
    --dataset_split="val_split" \
    --measure_time \
    --save_image

# Args
# --model_config  Config file corresponding to model. Required to build model.        
# --model_path    Path to the saved model
# --aug_configs   Directory of configs or config containing augmentation strategy to apply.
# --dataset_split train_split or val_split
# --output_path   Path to output logs and images
# --measure_time  Enable evaluation with time (fps) measurement | True by default     
# --save_image    Enable saving inference result image into out_img/ | False by default
# --update_bn     Reset and update BatchNorm running mean/var with entire dataset | False by default
# --no-bn_fusion  Disable performing batch norm fusion with convolutional layers | bn_fusion is enabled by default