import json
import os
import shutil
import re
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
data_number = 19
root_path = '/home/user/git_code/LMOE/utils/CTdataset/output'
# File path where the JSON files are located
folder_path = '/home/user/git_code/LMOE/utils/CTdataset/output/' + str(data_number) + '/'
folder_merge_path = folder_path + 'results/merge/'
if not os.path.exists(folder_merge_path):
    os.makedirs(folder_merge_path)
if not os.path.exists(folder_path + 'results/'):
    os.makedirs(folder_path + 'results/')
if not os.path.exists(folder_path + 'json/'):
    os.makedirs(folder_path + 'json/')
######################################################
#step1:将识别到黄框，同时识别到诊断信息的图片移动到对应的文件夹中
######################################################
def makefolder(path):
    class_name_position = ["leftlung_up", "leftlung_down", "rightlung_up", "rightlung_middle", "rightlung_down"]
    class_name_train = ["ground_truth", "test", "train"]
    class_name_series = ["highrisk", "mediumrisk"]
    class_path = []

    for position in class_name_position:
        # Create the top-level folder path
        top_level_path = folder_merge_path + position
        class_path.append(top_level_path)

        # Check if the top-level folder exists, create it if not
        if not os.path.exists(top_level_path):
            os.mkdir(top_level_path)

        for series in class_name_train:
            # Create the subfolder path
            subfolder_path = os.path.join(top_level_path, series)
            class_path.append(subfolder_path)
            # Check if the subfolder exists, create it if not
            if not os.path.exists(subfolder_path):
                os.mkdir(subfolder_path)
            if series == 'train':
                subfolder_path1 = os.path.join(subfolder_path, 'good')
                if not os.path.exists(subfolder_path1):
                    os.mkdir(subfolder_path1)
            if series == 'ground_truth':
                # Check if the subfolder exists, create it if not
                for series2 in class_name_series:
                    subfolder_path2 = os.path.join(subfolder_path, series2)
                    if not os.path.exists(subfolder_path2):
                        os.mkdir(subfolder_path2)
            if series == 'test':
                # Check if the subfolder exists, create it if not
                for series2 in class_name_series:
                    subfolder_path2 = os.path.join(subfolder_path, series2)
                    if not os.path.exists(subfolder_path2):
                        os.mkdir(subfolder_path2)
                subfolder_path2_1 = os.path.join(subfolder_path, 'good')
                if not os.path.exists(subfolder_path2_1):
                    os.mkdir(subfolder_path2_1)
            else:
                pass

def choosefold(data):
    contains_left = any('左' in value for value in data['info'])
    contains_right = any('右' in value for value in data['info'])
    contains_up = any('上' in value for value in data['info'])
    contains_middle = any('中' in value for value in data['info'])
    contains_down = any('下' in value for value in data['info'])
    contains_high_risk = any('高危' in value for value in data['info'])
    contains_medium_risk = any('中危' in value for value in data['info'])
    if contains_left and contains_up and contains_high_risk:
        classes = "leftlung_up/test/highrisk"
    elif contains_left and contains_down and contains_high_risk:
        classes = "leftlung_down/test/highrisk"
    elif contains_left and contains_up and contains_medium_risk:
        classes = "leftlung_up/test/mediumrisk"
    elif contains_left and contains_down and contains_medium_risk:
        classes = "leftlung_down/test/mediumrisk"
    elif contains_right and contains_up and contains_medium_risk:
        classes = "rightlung_up/test/mediumrisk"
    elif contains_right and contains_up and contains_high_risk:
        classes = "rightlung_up/test/highrisk"
    elif contains_right and contains_middle and contains_medium_risk:
        classes = "rightlung_middle/test/mediumrisk"
    elif contains_right and contains_middle and contains_high_risk:
        classes = "rightlung_middle/test/highrisk"
    elif contains_right and contains_down and contains_high_risk:
        classes = "rightlung_down/test/highrisk"
    else:
        classes = "rightlung_down/test/medium_risk"
    return classes

makefolder(folder_merge_path)
# List to store the contents of all JSON files
all_json_contents = []
# Iterate through each file in the directory
panduan = ['左','右','上','下','中','高危','中危']
data_temp = []
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        # Open and read the JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            # for line in file:
            for i, line in enumerate(file):
                if i == 0:
                    continue
                else:
                    pass
                try:
                    data = json.loads(line)
                    #./output/mp4/张有福 17071340/192_z.jpg 为data['img_path']的格式，请将最后一个/后面的内容读取出来
                    img_path_new = data['img_path'].split('/')[-1]  # 192_z.jpg
                    #如果data['info']中为空且data['border']中为空，则将data['img_path']中的图片移动到文件夹位置path
                    # if data['info'] != [] and data['border'] != []:
                    if (data['info'] != [] and data['border'] != []) or (data['info'] == [] and data['border'] != []):
                        #判断data['info']中是否包含‘左’字段和‘高危’字段
                        #尝试判断data['info']中是否为空
                        try:
                            data_temp = data['info']
                            classes = choosefold(data)
                            path = folder_merge_path + classes + '/' + img_path_new
                        except Exception as e:
                            data['info'] = data_temp
                            classes = choosefold(data)
                            path = folder_merge_path + classes + '/' + img_path_new
                        try:
                            shutil.move(data['img_path'], path)
                            data['img_path'] = path
                            all_json_contents.append(data)
                        except Exception as e:
                            print(f"error:{e}")
                            continue
                    else:
                        continue
                except json.JSONDecodeError:
                    print(f"Error reading the line in file {filename}")
                    continue

# Consolidate all JSON contents into one JSON object
consolidated_json = {
    "files": all_json_contents
}

# Define the path for the new consolidated JSON file
consolidated_json_path_all = folder_path + 'json/all.json'
# Write the consolidated JSON data to a new file
with open(consolidated_json_path_all, 'w', encoding='utf-8') as file:
    json.dump(consolidated_json, file, ensure_ascii=False, indent=4)
    # Write the consolidated JSON data to a new file

#########################################################
#step2:补全诊断信息为空的图片的json文件,并将其移动到good的文件夹中
#########################################################
all_json_contents_heal = []
#所有正常的图片的json文件
consolidated_heal_json_path = folder_path + 'json/all_heal.json'

data_all = []
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        # Open and read the JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            # for line in file:
            for i, line in enumerate(file):
                try:
                    data = json.loads(line)
                    data_all.append(data['info'])
                except Exception as e:
                    print(f"error:{e}")
                    continue
            # print(data_all)
        j = 0
        # 如果data_all字段中存在空的列表，则复制临近不是空的字段的信息
        for i, data1 in enumerate(data_all):
            # print("i is {},data is {}".format(i,data))
            if data_all[i] == []:
                # 复制前一个不为空的字段，如果前一个字段为空，则复制后一个不为空的字段，直到找到不为空的字段
                while data_all[i] == []:
                    if data_all[i - j] != []:
                        # print(data_all[i-j] )
                        data_all[i] = data_all[i - j]
                    elif data_all[i + j] != []:
                        # print(data_all[i+j])
                        data_all[i] = data_all[i + j]
                    j = j + 1
                j = 0
            else:
                continue
        with open(file_path, 'r', encoding='utf-8') as file:
            # for line in file:
            for i, line in enumerate(file):
                try:
                    data = json.loads(line)
                    data['info'] = data_all[i]
                    all_json_contents_heal.append(data)
                except Exception as e:
                    print(f"error:{e}")
                    continue

consolidated_json_heal = {
    "files": all_json_contents_heal
}

with open(consolidated_heal_json_path, 'w', encoding='utf-8') as file:
    json.dump(consolidated_json_heal, file, ensure_ascii=False, indent=4)

def choosefoldforgood(data):
    contains_left = any('左' in value for value in data['info'])
    contains_right = any('右' in value for value in data['info'])
    contains_up = any('上' in value for value in data['info'])
    contains_middle = any('中' in value for value in data['info'])
    contains_down = any('下' in value for value in data['info'])
    if contains_left and contains_up:
        classes = "leftlung_up/test/good"
    elif contains_left and contains_down :
        classes = "leftlung_down/test/good"
    elif contains_right and contains_up:
        classes = "rightlung_up/test/good"
    elif contains_right and contains_middle:
        classes = "rightlung_middle/test/good"
    elif contains_right and contains_down:
        classes = "rightlung_down/test/good"
    else:
        pass
    return classes

#将正常的图片移动到good文件夹中
# folder_good_path = '/home/snowolf/colmap_leaf/output/mp4/results/good/'
all_json_good = []
# for filename in os.listdir(folder_path):
filename = consolidated_heal_json_path
base_path = '/home/user/git_code/LMOE/utils/CTdataset/'
if filename.endswith('.json'):
    # Open and read the JSON file
    with open(filename, 'r', encoding='utf-8') as file:
        try:
            data_all = json.load(file)
            for i, data in enumerate(data_all['files']):
                #./output/mp4/张有福 17071340/192_z.jpg 为data['img_path']的格式，请将最后一个/后面的内容读取出来
                img_path_new = data['img_path'].split('/')[-1]  # 192_z.jpg
                #如果data['info']中为空且data['border']中为空，则将data['img_path']中的图片移动到文件夹位置path
                if data['info'] != [] and data['border'] == []:
                    #判断data['info']中是否包含‘左’字段和‘高危’字段
                    #尝试判断data['info']中是否为空
                    try:
                        data_temp = data['info']
                        classes = choosefoldforgood(data)
                        path = folder_merge_path + classes + '/' + img_path_new
                    except Exception as e:
                        data['info'] = data_temp
                        classes = choosefoldforgood(data)
                        path = folder_merge_path + classes + '/' + img_path_new
                    try:
                        img_path = data['img_path']
                        img2path = img_path.replace('./', base_path)
                        #合并路径'home/snowolf/colmap_leaf',data['img_path']
                        #/home/snowolf/colmap_leaf/output/mp4/张有福17071340/104.jpg
                        if os.path.exists(img2path):
                            path_fortrain = path.split('test')[0] + 'train/good/' + img_path_new
                            shutil.copy(img2path, path_fortrain)
                            shutil.move(img2path, path)
                            data['img_path'] = path
                            data['info'] = '无异常,' + classes.split('/')[0]
                            all_json_good.append(data)
                        else:
                            print("fail!")
                    except Exception as e:
                        print(f"error:{e}")
                        continue
                else:
                    continue
        except json.JSONDecodeError:
            print(f"Error reading the line in file {filename}")

consolidated_json_good = {
    "files": all_json_good
}

# Define the path for the new consolidated JSON file
consolidated_json_path_good = folder_path + 'json/all_good.json'
# Write the consolidated JSON data to a new file
with open(consolidated_json_path_good, 'w', encoding='utf-8') as file:
    json.dump(consolidated_json_good, file, ensure_ascii=False, indent=4)

#########################################################
# step3:制作训练用mask
#########################################################
#读取all.json文件后，针对每个图片，根据data['border']中的坐标信息，将图片中的肺部区域标记出来，然后将标记后的图片保存到对应的文件夹中
#首先读取data['img_path']中图片的路径，然后读取data['border']中的坐标信息，然后将图片中的肺部区域标记出来，然后将标记后的图片保存到对应的文件夹中
filename = consolidated_json_path_all
base_path = '/home/user/git_code/LMOE/utils/CTdataset/'
if filename.endswith('.json'):
    with open(filename, 'r', encoding='utf-8') as file:
        try:
            data_all = json.load(file)
            for i, data in enumerate(data_all['files']):
                path_parts = data['img_path'].split('/')
                risk_category = path_parts[-2] if len(path_parts) > 1 else None
                position = path_parts[-4] if len(path_parts) > 1 else None
                imgname = path_parts[-1].split('.')[0]
                #读取图片，按照图片大小生成rect大小的模板，模板内为1,其他为0
                img_path = data['img_path']
                if os.path.exists(img_path):
                    width, height = 1415, 849
                    # 创建一个纯黑色图像
                    mask = np.zeros((height, width, 3), dtype=np.uint8)
                    # 添加白色区域
                    border = data['border'][0]  # 假设只有一个边界
                    x1, y1, x2, y2 = map(int, [border['x1'], border['y1'], border['x2'], border['y2']])
                    mask[y1:y2, x1:x2] = [255, 255, 255]

                    #将图片保存到对应的文件夹中
                    if risk_category == 'highrisk':
                        if position == 'leftlung_up':
                            path = folder_merge_path + 'leftlung_up/ground_truth/highrisk/' + imgname + '_mask.jpg'
                        elif position == 'leftlung_down':
                            path = folder_merge_path + 'leftlung_down/ground_truth/highrisk/' + imgname + '_mask.jpg'
                        elif position == 'rightlung_up':
                            path = folder_merge_path + 'rightlung_up/ground_truth/highrisk/' + imgname + '_mask.jpg'
                        elif position == 'rightlung_middle':
                            path = folder_merge_path + 'rightlung_middle/ground_truth/highrisk/' + imgname + '_mask.jpg'
                        elif position == 'rightlung_down':
                            path = folder_merge_path + 'rightlung_down/ground_truth/highrisk/' + imgname + '_mask.jpg'
                        else:
                            pass
                    elif risk_category == 'mediumrisk':
                        if position == 'leftlung_up':
                            path = folder_merge_path + 'leftlung_up/ground_truth/mediumrisk/' + imgname + '_mask.jpg'
                        elif position == 'leftlung_down':
                            path = folder_merge_path + 'leftlung_down/ground_truth/mediumrisk/' + imgname + '_mask.jpg'
                        elif position == 'rightlung_up':
                            path = folder_merge_path + 'rightlung_up/ground_truth/mediumrisk/' + imgname + '_mask.jpg'
                        elif position == 'rightlung_middle':
                            path = folder_merge_path + 'rightlung_middle/ground_truth/mediumrisk/' + imgname + '_mask.jpg'
                        elif position == 'rightlung_down':
                            path = folder_merge_path + 'rightlung_down/ground_truth/mediumrisk/' + imgname + '_mask.jpg'
                        else:
                            pass
                    else:
                        pass
                    # 保存图片
                    cv2.imwrite(path, mask)
                else:
                    print("fail!")

        except json.JSONDecodeError:
            print(f"Error reading the line in file {filename}")
