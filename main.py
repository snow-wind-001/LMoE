import argparse
import sys
import os
import yaml
import torch
import logging
import numpy as np
import random
import time
import deepspeed
# from header import *
import importlib
#读取参数yaml文件
def load_config(path):
    config_path = path
    configuration = None
    try:
        with open(config_path) as f:
            configuration = yaml.load(f, Loader=yaml.FullLoader)
        print(f'[!] load base configuration: {config_path}')
    except Exception as e:
        print('{}导入参数文件失败!!'.format(config_path))
    return configuration


def args_prase(key):
    parser = argparse.ArgumentParser(description="配置参数解析")
    #####################
    #    不同任务的配置   #
    #####################
    # 1 训练Medicalimaging expert
    if key == '1':
        parser.add_argument('--root_dir', type=str, default='./')
        parser.add_argument('--model', type=str, default='OpenLLAMAPEFTModel')
        parser.add_argument('--model_conf', type=str, default='openllama_peft')
        parser.add_argument('--mode', type=str, default='train')
        parser.add_argument('--local_rank', default=0, type=int)
        parser.add_argument('--save_path', type=str, default='./checkpoints/Medicalimaging')
        parser.add_argument('--log_path', type=str, default='./ckpt/Medicalimaging/log/')
        parser.add_argument('--ds_config_path',type=str, default='./dsconfig/openllama_peft_stage_1.json')

        # model configurations
        parser.add_argument('--imagebind_ckpt_path', type=str,
                            default='/home/user/git_code/LMOE/code/AnomalyGPT/pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth')  # the path that stores the imagebind checkpoint
        parser.add_argument('--vicuna_ckpt_path', type=str,
                            default='/home/user/DISK/checkpoints/CT/baichuan-vicuna-7b/')  # the path that stores the vicuna checkpoint
        parser.add_argument('--delta_ckpt_path', type=str,
                            default='/home/user/git_code/LMOE/code/AnomalyGPT/pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt')  # the delta parameters trained in stage 1          parser.add_argument('--max_tgt_len', type=int, default=1024)  # the maximum sequence length
        parser.add_argument('--stage', type=int, default=1)  # the maximum sequence length
        parser.add_argument('--data_path', type=str,
                            default='/home/user/git_code/LMOE/code/AnomalyGPT/data/pandagpt4_visual_instruction_data.json')  # the maximum sequence length
        parser.add_argument('--image_root_path', type=str,
                            default='utils/CTdataset/output/')  # the maximum sequence length
        args = parser.parse_args()
        parser = vars(args)
        # print('parser[root] is {}'.format(parser['root_dir']))
        config = load_config(os.path.join(parser['root_dir'], 'config', 'base.yaml'))
        # print(config)
        parser.update(config)

    # 2 训练Oncologist expert
    elif key == '2':
        parser.add_argument('--model', type=str, default='LungsCancerMoE')
        parser.add_argument('--local_rank', default=0, type=int)
        parser.add_argument('--save_path', type=str, default='./checkpoints/train_lmoe')
        parser.add_argument('--log_path', type=str, default='./ckpt/train_lmoe/log_rest/')
        # model configurations
        parser.add_argument('--model_name_or_path', type=str,
                            default='/home/user/DISK/checkpoints/lmoe/ziya13bmerge')  # the path that stores the medical checkpoint
        parser.add_argument('--max_tgt_len', type=int, default=1024)  # the maximum sequence length
        parser.add_argument('--stage', type=int, default=1)  # the maximum sequence length
        # chatpdf
        parser.add_argument("--sim_model", type=str, default="/home/user/DISK/checkpoints/lmoe/text2vec")
        parser.add_argument("--gen_model_type", type=str, default="llama")
        parser.add_argument("--gen_model", type=str, default="/home/user/DISK/checkpoints/lmoe/ziya13bmerge")
        parser.add_argument("--lora_model", type=str, default=None)
        parser.add_argument("--corpus_files", type=str, default="/home/user/git_code/LMOE/knowledge_floder")
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--int4", action='store_false', help="use int4 quantization")
        parser.add_argument("--int8", action='store_true', help="use int8 quantization")
        parser.add_argument("--chunk_size", type=int, default=100)
        parser.add_argument("--chunk_overlap", type=int, default=5)

    # 3 训练多专家系统LMoE
    elif key == '3':
        parser.add_argument('--model', type=str, default='LungsCancerMoE')
        parser.add_argument('--local_rank', default=0, type=int)
        parser.add_argument('--save_path', type=str, default='./checkpoints/train_lmoe')
        parser.add_argument('--log_path', type=str, default='./ckpt/train_lmoe/log_rest/')
        # model configurations
        parser.add_argument('--imagebind_ckpt_path', type=str,
                            default='/home/user/git_code/LMOE/code/AnomalyGPT/pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth')  # the path that stores the imagebind checkpoint
        parser.add_argument('--vicuna_ckpt_path', type=str,
                            default='/home/user/DISK/checkpoints/CT/baichuan-vicuna-7b/')  # the path that stores the vicuna checkpoint
        parser.add_argument('--delta_ckpt_path', type=str,
                            default='/home/user/git_code/LMOE/code/AnomalyGPT/pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt')  # the delta parameters trained in stage 1
        parser.add_argument('--model_name_or_path', type=str,
                            default='/home/user/DISK/checkpoints/lmoe/ziya13bmerge')  # the path that stores the medical checkpoint

        parser.add_argument('--max_tgt_len', type=int, default=1024)  # the maximum sequence length
        parser.add_argument('--stage', type=int, default=1)  # the maximum sequence length
        parser.add_argument('--data_path', type=str,
                            default='/home/user/git_code/LMOE/code/AnomalyGPT/data/pandagpt4_visual_instruction_data.json')  # the maximum sequence length
        parser.add_argument('--image_root_path', type=str,
                            default='utils/CTdataset/output/')  # the maximum sequence length
        # chatpdf
        parser.add_argument("--sim_model", type=str, default="/home/user/DISK/checkpoints/lmoe/text2vec")
        parser.add_argument("--gen_model_type", type=str, default="llama")
        parser.add_argument("--gen_model", type=str, default="/home/user/DISK/checkpoints/lmoe/ziya13bmerge")
        parser.add_argument("--lora_model", type=str, default=None)
        parser.add_argument("--corpus_files", type=str, default="/home/user/git_code/LMOE/knowledge_floder")
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--int4", action='store_false', help="use int4 quantization")
        parser.add_argument("--int8", action='store_true', help="use int8 quantization")
        parser.add_argument("--chunk_size", type=int, default=100)
        parser.add_argument("--chunk_overlap", type=int, default=5)

    # 4 测试Oncologist expert
    elif key == '4':
        parser.add_argument('--model_type', default=None, type=str, required=True)
        parser.add_argument('--base_model', default=None, type=str, required=True)
        parser.add_argument('--lora_model', default="", type=str, help="If None, perform inference on the base model")
        parser.add_argument('--tokenizer_path', default=None, type=str)
        parser.add_argument('--template_name', default="vicuna", type=str,
                            help="Prompt template name, eg: alpaca, vicuna, baichuan, chatglm2 etc.")
        parser.add_argument("--repetition_penalty", type=float, default=1.0)
        parser.add_argument("--max_new_tokens", type=int, default=512)
        parser.add_argument('--data_file', default=None, type=str,
                            help="A file that contains instructions (one instruction per line)")
        parser.add_argument('--interactive', action='store_true',
                            help="run in the instruction mode (default multi-turn)")
        parser.add_argument('--single_tune', action='store_true', help='Whether to use single-tune model')
        parser.add_argument('--do_sample', action='store_true', help='Whether to use sampling in generation')
        parser.add_argument('--output_file', default='./predictions_result.jsonl', type=str)
        parser.add_argument("--eval_batch_size", type=int, default=4)
        parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
        parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
        parser.add_argument('--load_in_8bit', action='store_true', help='Whether to load model in 8bit')
        parser.add_argument('--load_in_4bit', action='store_true', help='Whether to load model in 4bit')

    # 5 测试Medicalimaging expert
    elif key == '5':
        # diff
        parser.add_argument('--model', type=str, default='LungsCancerMoE')
        parser.add_argument('--local_rank', default=0, type=int)
        parser.add_argument('--save_path', type=str, default='./checkpoints/train_lmoe')
        # model configurations
        parser.add_argument('--imagebind_ckpt_path', type=str,
                            default='/home/user/git_code/LMOE/code/AnomalyGPT/pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth')  # the path that stores the imagebind checkpoint
        parser.add_argument('--vicuna_ckpt_path', type=str,
                            default='/home/user/DISK/checkpoints/CT/baichuan-vicuna-7b/')  # the path that stores the vicuna checkpoint
        parser.add_argument('--delta_ckpt_path', type=str,
                            default='/home/user/git_code/LMOE/code/AnomalyGPT/pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt')  # the delta parameters trained in stage 1
        parser.add_argument('--max_tgt_len', type=int, default=1024)  # the maximum sequence length
        parser.add_argument('--stage', type=int, default=2)  # the maximum sequence length
        parser.add_argument('--data_path', type=str,
                            default='/home/user/git_code/LMOE/code/AnomalyGPT/data/pandagpt4_visual_instruction_data.json')  # the maximum sequence length
        parser.add_argument('--image_root_path', type=str,
                            default='utils/CTdataset/output/')  # the maximum sequence length

    # 6 测试向量数据库
    elif key == '6':
        parser.add_argument('--model', type=str, default='LungsCancerMoE')
        parser.add_argument('--local_rank', default=0, type=int)
        parser.add_argument('--save_path', type=str, default='./checkpoints/train_lmoe')
        # model configurations
        parser.add_argument('--imagebind_ckpt_path', type=str,
                            default='/home/user/git_code/LMOE/code/AnomalyGPT/pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth')  # the path that stores the imagebind checkpoint
        parser.add_argument('--vicuna_ckpt_path', type=str,
                            default='/home/user/DISK/checkpoints/CT/baichuan-vicuna-7b/')  # the path that stores the vicuna checkpoint
        parser.add_argument('--delta_ckpt_path', type=str,
                            default='/home/user/git_code/LMOE/code/AnomalyGPT/pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt')  # the delta parameters trained in stage 1
        parser.add_argument('--model_name_or_path', type=str,
                            default='/home/user/DISK/checkpoints/lmoe/ziya13bmerge')  # the path that stores the medical checkpoint

        parser.add_argument('--max_tgt_len', type=int, default=1024)  # the maximum sequence length
        parser.add_argument('--stage', type=int, default=1)  # the maximum sequence length
        parser.add_argument('--data_path', type=str,
                            default='/home/user/git_code/LMOE/code/AnomalyGPT/data/pandagpt4_visual_instruction_data.json')  # the maximum sequence length
        parser.add_argument('--image_root_path', type=str,
                            default='utils/CTdataset/output/')  # the maximum sequence length

        # chatpdf
        parser.add_argument("--sim_model", type=str, default="/home/user/DISK/checkpoints/lmoe/text2vec")
        parser.add_argument("--gen_model_type", type=str, default="llama")
        parser.add_argument("--gen_model", type=str, default="/home/user/DISK/checkpoints/lmoe/ziya13bmerge")
        parser.add_argument("--lora_model", type=str, default=None)
        parser.add_argument("--corpus_files", type=str, default="/home/user/git_code/LMOE/knowledge_floder")
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--int4", action='store_false', help="use int4 quantization")
        parser.add_argument("--int8", action='store_true', help="use int8 quantization")
        parser.add_argument("--chunk_size", type=int, default=100)
        parser.add_argument("--chunk_overlap", type=int, default=5)

    # 7 系统联调测试
    elif key == '7':
        parser.add_argument('--model', type=str, default='LungsCancerMoE')
        parser.add_argument('--local_rank', default=0, type=int)
        parser.add_argument('--save_path', type=str, default='./checkpoints/train_lmoe')
        # model configurations
        parser.add_argument('--imagebind_ckpt_path', type=str,
                            default='/home/user/git_code/LMOE/code/AnomalyGPT/pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth')  # the path that stores the imagebind checkpoint
        parser.add_argument('--vicuna_ckpt_path', type=str,
                            default='/home/user/DISK/checkpoints/CT/baichuan-vicuna-7b/')  # the path that stores the vicuna checkpoint
        parser.add_argument('--delta_ckpt_path', type=str,
                            default='/home/user/git_code/LMOE/code/AnomalyGPT/pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt')  # the delta parameters trained in stage 1
        parser.add_argument('--model_name_or_path', type=str,
                            default='/home/user/DISK/checkpoints/lmoe/ziya13bmerge')  # the path that stores the medical checkpoint

        parser.add_argument('--max_tgt_len', type=int, default=1024)  # the maximum sequence length
        parser.add_argument('--stage', type=int, default=1)  # the maximum sequence length
        parser.add_argument('--data_path', type=str,
                            default='/home/user/git_code/LMOE/code/AnomalyGPT/data/pandagpt4_visual_instruction_data.json')  # the maximum sequence length
        parser.add_argument('--image_root_path', type=str,
                            default='utils/CTdataset/output/')  # the maximum sequence length

        # chatpdf
        parser.add_argument("--sim_model", type=str, default="/home/user/DISK/checkpoints/lmoe/text2vec")
        parser.add_argument("--gen_model_type", type=str, default="llama")
        parser.add_argument("--gen_model", type=str, default="/home/user/DISK/checkpoints/lmoe/ziya13bmerge")
        parser.add_argument("--lora_model", type=str, default=None)
        parser.add_argument("--corpus_files", type=str, default="/home/user/git_code/LMOE/knowledge_floder")
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--int4", action='store_false', help="use int4 quantization")
        parser.add_argument("--int8", action='store_true', help="use int8 quantization")
        parser.add_argument("--chunk_size", type=int, default=100)
        parser.add_argument("--chunk_overlap", type=int, default=5)
    # args = parser.parse_args()
    # args_dict = vars(args)
    return parser

def initialize_distributed(args):
    args['master_ip'] = os.getenv('MASTER_ADDR', 'localhost')
    args['master_port'] = os.getenv('MASTER_PORT', '6000')
    args['world_size'] = int(os.getenv('WORLD_SIZE', '1'))
    args['local_rank'] = int(os.getenv('RANK', '0')) % torch.cuda.device_count()
    device = args['local_rank'] % torch.cuda.device_count()
    torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend='nccl')

def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def config_env(args):
    args['root_dir'] = './'
    config = load_config(os.path.join(args['root_dir'],'config',f'{args["model_conf"]}.yaml'))
    # print(config)
    if config is not None:  # 确保 config 不为 None
        args.update(config)
    else:
        # 可能需要处理 config 为 None 的情况
        print("警告: 无法加载配置文件，将使用默认配置。")
    #################################################
    # CPU调试屏蔽
    # initialize_distributed(args)
    #################################################
    set_random_seed(args['train']['seed'])


def build_directory(path):
    if os.path.exists(path):
        pass
    else: # recursively construct directory
        os.makedirs(path, exist_ok=True)

def main(parm):
    args = args_prase(parm)
    build_directory(args['save_path'])
    build_directory(args['log_path'])
    print(F'RUNNING MODE {args["mode"]}')
    if args['log_path']:
        logging.basicConfig(
            format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            level=logging.DEBUG,
            filename=f'{args["log_path"]}/train_{time.asctime()}.log',
            filemode='w'
        )
    config_env(args)
    # print(args['ds_config_path'])
    #################################################
    # CPU调试屏蔽
    # dschf = HfDeepSpeedConfig(args['ds_config_path'])
    # args['dschf'] = dschf
    #################################################
    #1.训练Medicalimaging expert
    if parm == '1':
        #打印运行参数
        print(f'已选择模型{parm}： Medicalimaging expert')
        print(f'模型参数如下 : \n{args}')
        os.makedirs(args['log_path'], exist_ok=True)
        train_Medicalimaging = importlib.import_module("libcode.model.train_Medicalimaging")
        train = getattr(train_Medicalimaging, 'train', None)
        train(args)
    #2.训练Oncologist expert
    elif parm == '2':
        print(f'已选择模型{parm}： vqgan')
        print(f'模型参数如下 : \n{args}')
        os.makedirs(args['log_path'], exist_ok=True)
        os.makedirs(args.weight_save_root, exist_ok=True)
        train_Medicalimaging = importlib.import_module("libcode.model.train_Oncologist")
        train = getattr(train_Medicalimaging, 'train', None)
        train(args)
    #3.训练多专家系统LMoE
    elif parm == '3':
        print(f'已选择模型{parm}： diffusion')
        print(f'模型参数如下 : \n{args}')
        print(f'创建权重保存路径 : {args.weight_save_root}')
        os.makedirs(args.weight_save_root, exist_ok=True)
        print(f'创建保存验证结果路径 : {args.results}')
        os.makedirs(args.weight_save_root + args.results, exist_ok=True)
        train_diffusion(args)
    #4.测试Oncologist expert
    elif parm == '4':
        print(f'已选择模型{parm}： gen_npy')
        print(f'模型参数如下 : \n{args}')
        gen_npy(args)
    #     TODO 5未编写
    #5.测试Medicalimaging expert
    elif parm == '5':
        print(f'已选择模型{parm}： gennerate')
        print(f'模型参数如下 : \n{args}')
        print(f'创建generate img 保存路径 : {args.generate_img}')
        os.makedirs(args.generate_img, exist_ok=True)
        generate(args)
    #6.测试向量数据库
    elif parm == '6':
        print(f'已选择模型{parm}： transfer')
        print(f'模型参数如下 : \n{args}')
        transfer(args)
    #7.系统联调测试
    elif parm == '7':
        print(f'已选择模型{parm}： llm generate')
        print(f'模型参数如下 : \n{args}')
        print(f'创建llm_generate_img 保存路径 : {args.llm_generate_img}')
        os.makedirs(args.llm_generate_img, exist_ok=True)
        llmgenerate(args)
    else:
        print('警告：！没有这个模型名字,请检查输入')
        print(f"congratulations，your {args.name}training is finish !")




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(f'shell: {sys.argv[0]} is running')
    parm = input(
        '请输入训练使用的模型:\n\r可选项为：'
        '|1.训练Medicalimaging expert\t'
        '|2.训练Oncologist expert\t'
        '|3.训练多专家系统LMoE\t'
        '|4.测试Oncologist expert\t'
        '|5.测试Medicalimaging expert\t'
        '|6.测试向量数据库:\t'
        '|7.系统联调测试\n\r\t')

    main(parm)


