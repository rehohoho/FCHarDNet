
python ../save_worst_performers.py \
    --dataset_path="<path-to-validation-logged-images-directory>" \
    --csv_path="<path-to-validation-logged-imagewise-metrics-csv-file>" \
    --target_metric="miou" \
    --perc=0.1