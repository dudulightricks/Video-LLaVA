

DATA_ROOT="/opt/msr-vtt-for-train"
CUDA_VISIBLE_DEVICES=0,1 python llava/train/train_mem.py \
    --model_name_or_path /opt/llava-video-7b \
    --version v1 \
    --data_path /opt/msr-vtt-for-train.json \
    --video_folder ${DATA_ROOT} \
    --image_folder ${DATA_ROOT} \
    --X "Video" "Image" \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --image_tower LanguageBind/LanguageBind_Image \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_x_start_end False \
    --mm_use_x_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/Video-LLaVA-7B \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "epoch" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --cache_dir "./cache_dir"
    --report_to wandb
