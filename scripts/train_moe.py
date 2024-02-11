import sys
sys.path.append('/home/user/git_code/LMOE')
from libcode.model import *
from header import *
from dataset import *
from config import *

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', type=str,default='LungsCancerMoE')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--save_path', type=str,default='./checkpoints/train_lmoe')
    parser.add_argument('--log_path', type=str,default='./ckpt/train_lmoe/log_rest/')
    # model configurations
    parser.add_argument('--imagebind_ckpt_path', type=str,default='/home/user/git_code/LMOE/code/AnomalyGPT_TEMP/pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth') # the path that stores the imagebind checkpoint
    parser.add_argument('--vicuna_ckpt_path', type=str,default='/home/user/DISK/checkpoints/CT/baichuan-vicuna-7b/') # the path that stores the vicuna checkpoint
    parser.add_argument('--delta_ckpt_path', type=str,default='/home/user/git_code/LMOE/code/AnomalyGPT_TEMP/pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt') # the delta parameters trained in stage 1
    parser.add_argument('--model_name_or_path', type=str,default='/home/user/DISK/checkpoints/lmoe/ziya13bmerge') # the path that stores the medical checkpoint

    parser.add_argument('--max_tgt_len', type=int,default=1024) # the maximum sequence length
    parser.add_argument('--stage', type=int,default=1) # the maximum sequence length
    parser.add_argument('--data_path', type=str,default='/home/user/git_code/LMOE/code/AnomalyGPT_TEMP/data/pandagpt4_visual_instruction_data.json') # the maximum sequence length
    parser.add_argument('--image_root_path', type=str,default='utils/CTdataset/output/') # the maximum sequence length

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
    args['ds_config_path'] = f'/home/user/git_code/LMOE/dsconfig/{args["model"]}_stage_{args["stage"]}.json'
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


    length = args['epochs'] * len(train_data) // args['world_size'] // dschf.config['train_micro_batch_size_per_gpu']
    total_steps =  args['epochs'] * len(train_data) // dschf.config['train_batch_size']
    args['total_steps'] = total_steps
    agent = load_model(args)
    torch.distributed.barrier()

    print('epochs is {}'.format(args['epochs']))

    # begin to train
    pbar = tqdm(total= 2 * length)    # maximum total number
    current_step = 0
    for epoch_i in tqdm(range(args['epochs'])):
        iter_every_epoch = 0
        for batch in train_iter:
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
