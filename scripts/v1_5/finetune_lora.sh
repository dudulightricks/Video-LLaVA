
IMAGE_FOLDER="/opt/dataset-mix-for-gemini-v2"
VIDEO_FOLDER="/opt/dataset-mix-for-gemini-v2"
cd /opt/Video-LLaVA
export PYTHONPATH=/opt/Video-LLaVA

/home/ubuntu/.pyenv/shims/deepspeed videollava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2_offload.json \
    --model_name_or_path /opt/Video-LLaVA/checkpoints/llava-video-gemini-dataset-mix-242k-on-1136k \
    --version v1 \
    --data_path /opt/dataset-mix-gemini-v2-train.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower LanguageBind/LanguageBind_Image \
    --video_folder ${VIDEO_FOLDER} \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-video-gemini-dataset-mix-242k-on-1136k-epoch3 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048  --tokenizer_model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --cache_dir "./cache_dir"
