import importlib
import os
import sys

# 检查模块是否存在
module_path = "../trainllm/supervised_finetuning.py"
if os.path.exists(module_path):
    print("The module `libcode.trainllm.supervised_finetuning` exists.")
else:
    print("The module `libcode.trainllm.supervised_finetuning` does not exist.")

# 正确地添加目录到sys.path
module_dir = os.path.dirname(module_path)  # 获取模块所在目录
if module_dir not in sys.path:
    sys.path.append(module_dir)

# 使用importlib导入模块并使用get_conv_template
supervised_finetuning = importlib.import_module("libcode.trainllm.supervised_finetuning")
get_conv_template = getattr(supervised_finetuning, 'get_conv_template', None)

if get_conv_template:
    print("Successfully imported `get_conv_template`.")
else:
    print("Failed to import `get_conv_template`.")
importlib.import_module('libcode.trainllm.supervised_finetuning')
from supervised_finetuning import get_conv_template