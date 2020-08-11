python train.py \
    --epochs 10 \
    --image_width 448 \
    --max_txt_length 20 \
    --batch_size 512 \
    --validate_every_step 300 \
    --model_name last_model.h5 \
    --data_directory dataset
    --pretrained_model snapshots_1108/snapshot-4.h5
