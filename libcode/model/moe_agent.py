from header import *


class DeepSpeedAgent:

    def __init__(self, model, args):
        super(DeepSpeedAgent, self).__init__()
        self.args = args
        self.model = model
        # 加载delta预训练参数
        # self.load_stage_1_parameters(args["delta_ckpt_path"])
        self.load_stage_1_parameters('/home/user/git_code/LMOE/libcode/AnomalyGPT_TEMP/pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt')

        for name, param in self.model.named_parameters():
            param.requires_grad = False

        # for name, param in self.model.image_decoder.named_parameters():
        #     param.requires_grad = True

        # for name, param in self.model.prompt_learner.named_parameters():
        #     param.requires_grad = True

        # load config parameters of deepspeed
        ds_params = json.load(open(self.args['ds_config_path']))
        ds_params['scheduler']['params']['total_num_steps'] = self.args['total_steps']
        ds_params['scheduler']['params']['warmup_num_steps'] = max(10, int(
            self.args['total_steps'] * self.args['warmup_rate']))
        self.ds_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config_params=ds_params,
            dist_init_required=True,
            args=types.SimpleNamespace(**args)
        )

    @torch.no_grad()
    def predict(self, batch):
        self.model.eval()
        string = self.model.generate_one_sample(batch)
        return string

    def train_model(self, batch, current_step=0, pbar=None):
        self.ds_engine.module.train()
        # acc ,总loss ，分类，综合专家结果，RAG专家结果
        mle_acc, loss, cls, out_result, response= self.ds_engine(batch)

        self.ds_engine.backward(loss)
        self.ds_engine.step()
        pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}')
        pbar.update(1)
        if self.args['local_rank'] == 0 and self.args['log_path'] and current_step % self.args['logging_step'] == 0:
            elapsed = pbar.format_dict['elapsed']
            rate = pbar.format_dict['rate']
            remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
            remaining = str(datetime.timedelta(seconds=remaining))
            logging.info(
                f'[!] progress: {round(pbar.n / pbar.total, 5)}; remaining time: {remaining}; loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}')

        mle_acc *= 100
        return mle_acc

    def save_model(self, path, current_step):
        # only save trainable model parameters
        # 只保存可训练模型参数的状态字典
        state_dict = {k: v.cpu() for k, v in self.ds_engine.module.state_dict().items() if v.requires_grad}
        torch.save(state_dict, f'{path}/pytorch_model.pt')
        # torch.save(checkpoint.state_dict(), f'{path}/pytorch_model.pt')
        # save tokenizer
        self.model.llama_tokenizer.save_pretrained(path)
        # save configuration
        self.model.llama_model.config.save_pretrained(path)
        print(f'[!] save model into {path}')

    def load_stage_1_parameters(self, path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        # 假设状态字典保存在 checkpoint 的 'state_dict' 键中
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # 如果不是，则假设整个 checkpoint 就是状态字典
            state_dict = checkpoint
        try:
            self.model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Error loading state dict: {e}")
            print("State dict keys:", state_dict.keys())