python train.py \
    --epochs 40 \
    --image_width 320 \
    --max_txt_length 42 \
    --batch_size 128 \
    --validate_every_step 10 \
    --model_name last_model.h5 \
    --pretrained_model snapshots/snapshot-2.h5
