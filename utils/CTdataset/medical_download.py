from huggingface_hub import login
from huggingface_hub import HfFolder
from datasets import load_dataset
HfFolder.save_token()
# 选择并加载特定配置的数据集
config_name = 'finetune'  # 选择 'finetune', 'pretrain' 或 'reward' 其中之一
dataset = load_dataset("shibing624/medical", config_name)

# 打印数据集信息
print(dataset)

# 打印训练集的第一个样本
print(dataset['train'][0])

# 将数据集保存到本地磁盘
dataset.save_to_disk("./dataset")