import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import libcode.model.openllama
from libcode.model.openllama import *
from libcode.model.moe import *
from libcode.model.Oncologist_llm import *
from libcode.model.chatpdf_oncolgist import *
from libcode.model.openllama import OpenLLAMAPEFTModel
class HistoryBuffer:
    """
定义历史缓冲区
history_buffer = HistoryBuffer()

# 添加一些历史记录
history_buffer.add_user_input("你好，我想了解历史。")
history_buffer.add_model_response("你好！关于历史，你有什么具体问题吗？")

# 现在清空历史记录
history_buffer.clear_history()

# 验证历史是否已清空
print(history_buffer.get_combined_history())  # 应该输出空字符串"""
    def __init__(self):
        self.user_inputs = []
        self.model_responses = []

    def add_user_input(self, input_text):
        self.user_inputs.append(input_text)

    def add_model_response(self, response_text):
        self.model_responses.append(response_text)

    def get_combined_history(self):
        # 合并用户输入和模型回应
        combined_history = ""
        for i in range(len(self.user_inputs)):
            combined_history += "User: " + self.user_inputs[i] + "\n"
            if i < len(self.model_responses):
                combined_history += "Model: " + self.model_responses[i] + "\n"
        return combined_history

    def clear_history(self):
        # 清空历史记录
        self.user_inputs = []
        self.model_responses = []


class LungsCancerMoE(nn.Module):
    '''LoRA for LLaMa model'''
    def __init__(self, args):
        super(LungsCancerMoE, self).__init__()

        self.args = args
        # ano
        self.Oncologist_Expert = OpenLLAMAPEFTModel(args=self.args)
        # langchain+pdf
        self.sim_model = BertSimilarity(model_name_or_path="/home/user/DISK/checkpoints/lmoe/text2vec",)
        self.m = ChatPDF(
            similarity_model=self.sim_model,
            generate_model_type="llama",
            generate_model_name_or_path="/home/user/DISK/checkpoints/lmoe/ziya13bmerge",
            lora_model_name_or_path=None,
            device=None,
            int4=True,
            int8=False,
            chunk_size=100,
            chunk_overlap=5,
            corpus_files="/home/user/git_code/LMOE/knowledge_floder".split(','),
        )
        # 不带chain
        # self.Oncologist_Expert = Oncologist_Model(args=self.args)
        # moe
        self.LungCancerGenetic_Expert = MoE(num_experts=2, x1dim=4096, x2dim=5120)
        # 定义历史缓冲区
        self.history_buffer = HistoryBuffer()
        self.query = [
        "病理诊断是否有肺癌?",
        " CT检测结果是什么?",
        "EGFR Exon19是否基因突变？",
        "EGFR Exon21是否基因突变？",
        "请给根据以上诊断结果，说明该肺癌由哪个基因突变引起？"
    ]

    def Medicalimaging_Expert_forward(self, inputs):
        loss , gen_acc, hidden_state = self.Oncologist_Expert(inputs)

    def Medicalimaging_Expert_generate(self, inputs):
        self.Medicalimaging_Expert.eval()
        loss, gen_acc, hidden_state1 = self.Medicalimaging_Expert.generate(inputs)


    # langchain
    def Oncologist_Expert_forward(self, inputs):
        inputs_text = inputs['text']
        self.Oncologist_Expert.eval()
        with torch.no_grad():
            # 此处未完成应五轮对话
            for i in range(5):
                output, hidden_state0 = self.Oncologist_Expert(inputs_text)
                self.history_buffer.add_user_input(inputs_text)
                self.history_buffer.add_model_response(output)
            print(output)
        return output

    def Oncologist_Expert_generate(self, inputs):
        inputs_text = inputs['text']
        self.Oncologist_Expert.eval()
        history = []
        with torch.no_grad():
            # 此处未完成应五轮对话
            output, hidden_state0 = self.Oncologist_Expert(inputs_text)
            print(output)
        return output


    def forward(self,inputs):
        inputs_text = inputs['text']
        label_ids = self.Oncologist_Expert.tokenizer.encode(inputs['output'])
        with torch.no_grad():
            response, reference_results, hidden_state0 = self.m.predict(self.query)
            # output, hidden_state0 = self.Oncologist_Expert(inputs_text)
            print(f'肿瘤专家意见{response}')
            loss, gen_acc,hidden_state1 = self.Medicalimaging_Expert(inputs)
            print(f'影像专家意见{response}')

        moe_output, cls= self.LungCancerGenetic_Expert(hidden_state0,hidden_state1)
        out_result = self.Oncologist_Expert.tokenizer.batch_decode(moe_output.to('cuda:1'), skip_special_tokens=True)[0]
        output_ids =  self.Oncologist_Expert.tokenizer.encode(out_result.to('cuda:1'))

        print(f'影像学专家准确率是:{gen_acc} 输出结果：{out_result} 多专家综合分类结果：{cls}')

        return gen_acc, loss, cls,label_ids, output_ids



    def generate(self, inputs):
        self.Medicalimaging_Expert.eval()
        self.Oncologist_Expert.eval()
        self.LungCancerGenetic_Expert.eval()

        inputs_text = inputs['text']
        with torch.no_grad():
            # 此处未完成应五轮对话
            output, hidden_state0 = self.Oncologist_Expert(inputs_text)
        output_text, pixel_output, hidden_state1 = self.Medicalimaging_Expert.generate(inputs)

        moe_output, cls = self.LungCancerGenetic_Expert(hidden_state0, hidden_state1)
        with torch.no_grad():
            out_result = self.Oncologist_Expert.tokenizer.batch_decode(moe_output, skip_special_tokens=True)[0]
        print(f'肿瘤专家意见{output}, 影像学专家意见{output_text} 多专家判定结果：{out_result} 多专家综合分类结果：{cls}')
        # print('shuchu1', output, hidden_state.shape)
        return output, output_text, out_result, cls

# def parser_args():
#     parser = argparse.ArgumentParser(description='train parameters')
#     parser.add_argument('--model', type=str)
#     parser.add_argument('--local_rank', default=0, type=int)
#     parser.add_argument('--save_path', type=str)
#     parser.add_argument('--log_path', type=str)
#     # model_name_or_path
#     # model configurations
#     parser.add_argument('--imagebind_ckpt_path','' ,type=str) # the path that stores the imagebind checkpoint
#     parser.add_argument('--vicuna_ckpt_path', type=str) # the path that stores the vicuna checkpoint
#     parser.add_argument('--delta_ckpt_path', type=str) # the delta parameters trained in stage 1
#     parser.add_argument('--max_tgt_len', type=int) # the maximum sequence length
#     parser.add_argument('--stage', type=int) # the maximum sequence length
#     parser.add_argument('--data_path', type=str) # the maximum sequence length
#     parser.add_argument('--image_root_path', type=str) # the maximum sequence length
#
#     return parser.parse_args()
#
#
# args = {
#     'model': 'openllama_peft',
#     'OEpath':'/home/user/DISK/checkpoints/lmoe/ziya13bmerge',
#     'imagebind_ckpt_path':'/home/user/git_code/LMOE/pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
#     'vicuna_ckpt_path':'/home/user/git_code/LMOE/pretrained_ckpt/vicuna_ckpt/7b_v0',
#     'anomalygpt_ckpt_path': '/home/user/git_code/LMOE/pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt',
#     'delta_ckpt_path': '/home/user/git_code/LMOE/pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt',
#     'max_tgt_len': 512,
#     'model_name_or_path':"/home/user/DISK/checkpoints/lmoe/ziya13bmerge",
#     'stage':1,
#     'lora_r': 32,
#     'lora_alpha': 32,
#     'lora_dropout': 0.1
#     }

#
# import torchvision.transforms as transforms
# from PIL import Image
#
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # 例如，调整图像大小（如果需要）
#     transforms.ToTensor()
# ])
# x=[]
# image_paths = '/home/user/git_code/LMOE/03136042_590.jpg'
# images = transform(Image.open(image_paths).convert('RGB'))
# images = images.unsqueeze(0)
# inputs = {'text': '你好',
#           'output': '不是很好',
#           'max_tgt_len': 1024,
#           'top_p': 1,
#           'temperature': 1,
#           'prompt': '你好',
#           'modality_cache':'./cache',
#           'image_paths': image_paths,
#           # 'masks':,
#          'output_texts':'HAHAHAHAH',
#           'images':[images],}
# # args.OEpath = "/home/user/DISK/checkpoints/lmoe/ziya13bmerge"
# model = LungsCancerMoE(args)
# out = model.Oncologist_Expert_forward(inputs)
# # gen_acc, loss, cls,label_ids, output_ids = model(inputs)
# print('cls')
