#!/bin/bash

deepspeed --include localhost:0,1,2 --master_port 28400 train_ano.py \
    --model openllama_peft \
    --stage 1\
    --imagebind_ckpt_path /home/user/git_code/LMOE/code/AnomalyGPT/pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth\
    --vicuna_ckpt_path /home/user/DISK/checkpoints/CT/baichuan-vicuna-7b/\
    --delta_ckpt_path /home/user/git_code/LMOE/code/AnomalyGPT/pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt\
    --max_tgt_len 1024\
    --data_path  /home/user/git_code/LMOE/code/AnomalyGPT/data/pandagpt4_visual_instruction_data.json\
    --image_root_path /home/user/git_code/LMOE/utils/CTdataset/output\
    --save_path  ./checkpoints/train_ano/\
    --log_path ./checkpoints/log_rest/
