from dataset import *
from config import *
from header import *
from libcode.model import *



def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', default='openllama_peft', type=str)
    # parser.add_argument('--model', default='LungsCancerMoE', type=str)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--save_path', default='./checkpoints/train_ano/', type=str)
    parser.add_argument('--log_path', default='./checkpoints/log_rest/', type=str)
    # model configurations
    # parser.add_argument('--imagebind_ckpt_path', type=str) # the path that stores the imagebind checkpoint
    # parser.add_argument('--vicuna_ckpt_path', type=str) # the path that stores the vicuna checkpoint
    # parser.add_argument('--delta_ckpt_path', type=str) # the delta parameters trained in stage 1
    # parser.add_argument('--model_name_or_path', type=str) # the path that stores the medical checkpoint
    # 医疗chatbot配置

    parser.add_argument('--imagebind_ckpt_path', default='/home/user/git_code/LMOE/libcode/AnomalyGPT_TEMP/pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth', type=str) # the path that stores the imagebind checkpoint
    parser.add_argument('--vicuna_ckpt_path', default='/home/user/DISK/checkpoints/CT/baichuan-vicuna-7b/', type=str) # the path that stores the vicuna checkpoint
    parser.add_argument('--delta_ckpt_path', default='/home/user/git_code/LMOE/libcode/AnomalyGPT_TEMP/pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt', type=str) # the delta parameters trained in stage 1
    parser.add_argument('--model_name_or_path', type=str) # the path that stores the medical checkpoint



    parser.add_argument('--max_tgt_len', type=int) # the maximum sequence length
    parser.add_argument('--stage', default=1, type=int) # the maximum sequence length
    parser.add_argument('--data_path', default='/home/user/git_code/LMOE/dataset/data_sft/pandagpt4_visual_instruction_data.json', type=str) # the maximum sequence length
    parser.add_argument('--image_root_path', default='/home/user/git_code/LMOE/utils/CTdataset/output', type=str) # the maximum sequence length

    return parser.parse_args()

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
    args['root_dir'] = '../'
    args['mode'] = 'train'
    config = load_config(args)
    # print(config)
    args.update(config)
    initialize_distributed(args)
    set_random_seed(args['seed'])

def build_directory(path):
    if os.path.exists(path):
        pass
    else: # recursively construct directory
        os.makedirs(path, exist_ok=True)

def main(**args):
    config_env(args)
    args['ds_config_path'] = f'dsconfig/{args["model"]}_stage_{args["stage"]}.json'
    dschf = HfDeepSpeedConfig(args['ds_config_path'])
    args['dschf'] = dschf

    build_directory(args['save_path'])
    build_directory(args['log_path'])

    if args['log_path']:
        logging.basicConfig(
            format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', 
            level=logging.DEBUG,
            filename=f'{args["log_path"]}/train_{time.asctime()}.log',
            filemode='w'
        )
    
    train_data, train_iter, sampler = load_mvtec_dataset(args)
    train_data_sft, train_iter_sft, sampler = load_sft_dataset(args)

    length = args['epochs'] * len(train_data) // args['world_size'] // dschf.config['train_micro_batch_size_per_gpu']
    total_steps = 2 * args['epochs'] * len(train_data) // dschf.config['train_batch_size']
    args['total_steps'] = total_steps
    agent = load_model(args)
    torch.distributed.barrier()

    # begin to train
    pbar = tqdm(total= 2 * length)    # maximum total number
    current_step = 0
    for epoch_i in tqdm(range(args['epochs'])):
        iter_every_epoch = 0
        for batch, batch_sft in zip(train_iter,train_iter_sft):
            iter_every_epoch += 1
            agent.train_model(
                batch, 
                current_step=current_step, 
                pbar=pbar
            )
            del batch
            current_step += 1
            # torch.cuda.empty_cache()
            # if iter_every_epoch % 1000 == 0:
            #     agent.save_model(args['save_path'], 0)
        # save at the end of the training
        torch.distributed.barrier()
        agent.save_model(args['save_path'], 0)

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    args['layers'] = [7,15,23,31]
    main(**args)
