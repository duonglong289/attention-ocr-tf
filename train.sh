python train.py \
    --epochs 40 \
    --image_width 320 \
    --max_txt_length 20 \
    --batch_size 256 \
    --validate_every_step 250 \
    --model_name last_model.h5 \
    --data_directory dataset
    # --pretrained_model snapshots/snapshot-2.h5
