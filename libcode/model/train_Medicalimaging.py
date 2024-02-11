from dataset import *
from config import *
# from header import *
from libcode.model import *
def train(args):
    train_data, train_iter, sampler = load_mvtec_dataset(args)
    train_data_sft, train_iter_sft, sampler = load_sft_dataset(args)
    dschf = args['dschf']
    length = args['epochs'] * len(train_data) // args['world_size'] // dschf.config['train_micro_batch_size_per_gpu']
    total_steps = 2 * args['epochs'] * len(train_data) // dschf.config['train_batch_size']
    args['total_steps'] = total_steps
    agent = load_model(args)
    torch.distributed.barrier()

    # begin to train
    pbar = tqdm(total=2 * length)  # maximum total number
    current_step = 0
    for epoch_i in tqdm(range(args['epochs'])):
        iter_every_epoch = 0
        for batch, batch_sft in zip(train_iter, train_iter_sft):
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