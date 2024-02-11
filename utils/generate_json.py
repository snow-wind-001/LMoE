#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import shutil
import random

# 创建一个空的字典列表
data = []

# class name
cls_name = ['crack', 'scratch', 'shatter', 'decay', 'oxidation']


# 文件目录：
# ROOT___A__catagery0__catagery0
#        |__catagery1
#        |__catagery2
#        |__catagery3
#        |__catagery4
#     |__B__catagery0
#        |__catagery1
#        |__catagery2
#        |__catagery3
#        |__catagery4
# l
abrootdir = '/home/user/DISK/datasets/cherygpt/mtecv'


i = 0
for ab_dir in os.listdir(abrootdir):
    ab_root = os.path.join(abrootdir, ab_dir)
    for catagery in os.listdir(ab_root):
        catadir = os.path.join(ab_root,catagery)
        # de dao lei bie
        for root,dirs,files in os.walk(catadir):
            for file in files:

                if file.endswith('.jpg') and 'mask'not in file:
                    i += 1
                    file_path = os.path.join(root,file)
                    # 图片获取
                    # 种类获取(从原图片路径分离)
                    Category = file_path.split('/')[-4]
                    if Category in cls_name:
                        Category = Category
                    else:
                        print('类不在范围内')

                    # 判断是否异常
                    if 'test' in file_path and 'frequency' in file_path:
                        isabnormal = True
                    else:
                        isabnormal = False
                    if isabnormal:
                        abnomal = 'Yes'
                        isorno = 'something'
                    else:
                        abnomal = 'No'
                        isorno = 'nothing'

                    # 判断是否是B工况 5khz与否
                    if ab_root == "B":
                        k50 = 'B'
                    else:
                        k50 = 'A'

                    # 图片改名
                    new_filename = Category + k50 + f'{i}.jpg'
                    new_filepath = '/home/user/git_code/cometition/AnomalyGPT/data/images/' + new_filename
                    # 使用shutil库进行文件复制和改名操作
                    shutil.copy(file_path, new_filepath)
                    # 是否异常获取
                    print(Category)
                    data.append({
                        "image_name": new_filename,
                        "conversation": [
                        {"from": "human", "value": f"""This image consists of four pictures arranged in a tic-tac-toe pattern and corresponds to one of the situations: 'crack,' 'scratch,' 'shatter,' 'decay,' or 'oxidation.'The top-left picture represents the time-frequency plot of the axial direction vibration signal at the bearing end. The top-right picture shows the time-frequency plot of the radial direction vibration signal at the bearing end. The bottom-left picture displays the time-frequency plot of the tangential direction vibration signal at the bearing end, while the bottom-right picture represents the time-frequency plot of the microphone signal.Next, I need to pose several questions regarding this image.Do you understand what I mean?"""},
                        {"from": "gpt", "value": f"Yes,I know,Please pose questions regarding this image."},
                        {"from": "human", "value": "Which category does this picture belong to?"},
                        {"from": "gpt", "value": f"{Category}"},
                        {"from": "human", "value": f"Is there anything abnormal about this picture?."},
                        {"from": "gpt", "value": f"{abnomal}, there is {isorno} wrong with this picture."},
                        {"from": "human", "value": f"What kind of conditions is this image close to?"},
                        {"from": "gpt", "value": f"this picture is close to condition {k50}."},
                        {"from": "human", "value": f"What kind of conditions is this image close to?"},
                        {"from": "gpt", "value": f"this picture is close to condition {k50}."},
                    ]
                    })

shuffle_list = data.copy()
random.shuffle(shuffle_list)
# 将数据保存到JSON文件
with open('/home/user/git_code/cometition/AnomalyGPT/data/pandagpt4_visual_instruction_data.json', 'w') as json_file:
    json.dump(shuffle_list, json_file, indent=4)

print("JSON 文件已生成。")