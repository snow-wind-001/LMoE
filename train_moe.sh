#!/bin/bash

deepspeed --include localhost:0,1,2 --master_port 28400 train_moe.py \
    --model LungsCancerMoE \
    --stage 1\
    --sim_model /home/user/DISK/checkpoints/lmoe/text2vec\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth\
    --vicuna_ckpt_path /home/user/DISK/checkpoints/CT/baichuan-vicuna-7b/\
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt\
    --model_name_or_path /home/user/DISK/checkpoints/lmoe/ziya13bmerge\
    --max_tgt_len 1024\
    --data_path  ../data/pandagpt4_visual_instruction_data.json\
    --image_root_path utils/CTdataset/output/\
    --save_path  ./checkpoints/train_lmoe\
    --log_path ./ckpt/train_lmoe/log_rest/
