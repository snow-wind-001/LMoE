import torch
import sys
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer,AutoConfig

class Oncologist_Model(nn.Module):
    '''LoRA for LLaMa model'''
    def __init__(self, **args):
        super(Oncologist_Model, self).__init__(
        )
        config_kwargs = {
            "trust_remote_code": None,
            "cache_dir": None,
            # "revision": 'main',
            "use_auth_token": None,
            "output_hidden_states": True
        }
        # device = 'cuda:0'
        model_name_or_path = args['model_name_or_path']
        config = AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)
        self.model = LlamaForCausalLM.from_pretrained(model_name_or_path, device_map='auto', config=config)
        # self.model.quantize(4)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        si  = input()


    def generate_prompt(self,instruction):
        return f"""下面是描述任务的指令。编写适当完成请求的响应.\n\n### Instruction:{instruction}\n### Response: """

    def forward(self,inputs):
        q = self.generate_prompt(inputs)
        inputs = self.tokenizer.encode(q, return_tensors="pt")
        generate_ids = self.model.generate(
            inputs,
            max_new_tokens=120,
            do_sample=True,
            top_p=0.85,
            temperature=1.0,
            repetition_penalty=1.0
        )
        embedding = self.model(inputs, output_hidden_states=True)
        hidden_state = embedding.hidden_states
        hidden_state = torch.mean(hidden_state[0].squeeze(0) + hidden_state[-1].squeeze(0), axis=0)
        hidden_state = hidden_state.unsqueeze(0)
        output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]

        return output, hidden_state
# Oncologist_Expert = Oncologist_Model(model_name_or_path="/home/user/DISK/checkpoints/lmoe/ziya13bmerge")
# for i in range(50):
#     inputs = input('user')
#     output, hidden_state = Oncologist_Expert(inputs)
#     print('assitant',output)