import os
from torch.utils.data import Dataset
import cv2
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

import sys



describles = {}
describles['leftlung_up'] = "左肺上叶：位于左侧胸腔上前部，通过垂直的主裂缝与下叶分隔。其海绵状结构在CT图像上较为明亮，由多个上叶支气管分支供气。"
describles['leftlung_down'] = "位于左侧胸腔下部，紧邻膈肌，主裂缝将其与上叶分开。相比上叶，其肺泡密集度更高，形成典型的肺组织纹理"
describles['rightlung_up'] = "位于右侧胸腔下部，由横裂和斜裂与中叶及上叶分隔。在CT图像上呈现密集的海绵状结构，气管分支较为复杂。"
describles['rightlung_middle'] = "相对较小，位于右肺的前中部，两条裂缝将其与上下叶分开。其结构紧凑，纹理在CT图像上清晰可辨。"
describles['rightlung_down'] = "是右肺中最大的叶，覆盖胸腔的上前部。由斜裂和横裂与其他叶分隔，其海绵状结构在CT图像上表现为特定纹理。"


class LMoEDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.image_position = {"leftlung_up":'左肺上叶前段', "leftlung_down":'左肺下叶底段', "rightlung_up":'右肺上叶前段',
                               "rightlung_middle":'右肺中叶前段', "rightlung_down":'右肺下叶底段'}
        self.all_good_paths = []
        self.all_paths = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if "json" in file_path and 'all_good.json' in file:
                    self.all_good_paths.append(file_path)
                if "json" in file_path and 'all.json' in file:
                    self.all_paths.append(file_path)

        if transform is not None:
            self.transform = transform
        # self.transform = transform
        self.transform = transforms.Resize(
                                (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
                            )
        
        self.norm_transform = transforms.Compose(
                            [
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=(0.48145466, 0.4578275, 0.40821073),
                                    std=(0.26862954, 0.26130258, 0.27577711),
                                ),
                            ]
                        )

        self.paths = []
        self.x = []
        self.img_name = []
        self.img_path = []
        self.img_postion_key = []
        self.position = []
        self.risk = []
        self.discription = []
        self.DNA = []
        self.__read_json_all()

        # for root, dirs, files in os.walk(root_dir):
        #     for file in files:
        #         file_path = os.path.join(root, file)
        #         if "train" in file_path and "good" in file_path and 'jpg' in file:
        #             self.paths.append(file_path)
        #             self.x.append(self.transform(Image.open(file_path).convert('RGB')))
        #
        # self.prev_idx = np.random.randint(len(self.paths))

    def __len__(self):
        return len(self.paths)

    def __read_json_all(self):
        #读取self.all_paths中的json文件
        for json_path in self.all_paths:
            with open(json_path, 'r', encoding='utf-8') as file:
                try:
                    data_all = json.load(file)
                    for i, data in enumerate(data_all['files']):
                        self.img_name.append(data['img_path'].split('/')[-1])  # 192_z.jpg
                        self.img_path.append(data['img_path'])
                        self.img_postion_key.append(data['img_path'].split('merge')[-1].split('/')[1])
                        self.position.append(self.image_position[data['img_path'].split('merge')[-1].split('/')[1]])
                        self.risk.append(data['img_path'].split('/')[-2])
                        self.DNA.append(data['class'])
                        if '实性' in data['info']:
                            self.discription.append('实性')
                        else:
                            self.discription.append('磨玻璃')

                except json.JSONDecodeError:
                    print(f"Error reading the line in file {self.img_path}")
                    continue
        for json_path in self.all_good_paths:
            with open(json_path, 'r', encoding='utf-8') as file:
                try:
                    data_good = json.load(file)
                    for i, data in enumerate(data_good['files']):
                        self.img_name.append(data['img_path'].split('/')[-1])  # 192_z.jpg
                        self.img_path.append(data['img_path'])
                        self.img_postion_key.append(data['img_path'].split('merge')[-1].split('/')[1])
                        self.position.append(self.image_position[data['img_path'].split('merge')[-1].split('/')[1]])
                        self.discription.append("null")
                        self.risk.append("null")
                        self.DNA.append("null")
                except json.JSONDecodeError:
                    print(f"Error reading the line in file {self.img_path}")
                    continue



    def __getitem__(self, index):
        self.img_path_one = self.img_path[index]
        self.x = self.transform(Image.open(self.img_path[index]).convert('RGB'))
        self.class_name = self.img_postion_key[index]
        self.x = np.asarray(self.x)
        self.origin = self.x
        self.origin = self.norm_transform(self.origin)
        #由文件名定义mask路径
        # temp_path = self.img_path[index].replace('test', 'ground_truth')
        self.mask_path = self.img_path[index].replace('test', 'ground_truth').split('.jpg')[0] + '_mask.jpg'
        self.mask = np.asarray(self.transform(Image.open(self.mask_path)))
        conversation_abnormal = []
        conversation_abnormal.append(
            {"from": "human", "value": describles[self.img_postion_key[index]] +  "请针对这张肺CT片，结合医疗影像学知识，回答以下问题：" +
                                       "1,照片部位;2,纯磨玻璃还是实性;3,危险程度，4,基因突变点位。"})

        conversation_abnormal.append({"from": "gpt", "value": "好的，1,照片位于" + self.position[index] +
                                                            ";2,病变显示为"+ self.discription[index] +
                                                            ";3,病变危险等级为"+ self.risk[index]+"，4,基因突变点位为" + self.DNA[index]})

        conversation_normal = []
        conversation_normal.append(
            {"from": "human", "value": describles[self.img_postion_key[index]] +  "请针对这张肺CT片，结合医疗影像学知识，回答以下问题：" +
                                       "1,照片部位;2,纯磨玻璃还是实性;3,危险程度，4,基因突变点位。"})
        conversation_normal.append({"from": "gpt", "value": "好的，1,照片位于" + self.position[index] +
                                                              ";2,病变显示为" + self.discription[index] +
                                                              ";3,病变危险等级为" +
                                                                  self.risk[index] + "，4,基因突变点位为" + self.DNA[index]})


        return self.origin, conversation_normal, self.x, conversation_abnormal, self.class_name, self.mask, self.img_path_one, self.DNA[index], f"综合以上相关内容，诊断出以下相关结论：1.该患者肺癌肿瘤位于 {self.position[index]} ;" +f"2,病变显示为 {self.discription[index]} ;"+ f"3,病变危险等级为{self.risk[index] };" + f"4,基因突变点位为{self.DNA[index]}"



    def collate(self, instances):

        images = []
        texts = []
        class_names = []
        masks = []
        img_paths = []
        cls = []
        targets_res = []
        for instance in instances:
            images.append(instance[0])
            texts.append(instance[1])
            class_names.append(instance[4])
            masks.append(torch.zeros_like(instance[5]))
            img_paths.append(instance[6])

            images.append(instance[2])
            texts.append(instance[3])
            masks.append(instance[5])
            img_paths.append(instance[6])
            cls.append(instance[7])
            targets_res.append(instance[8])

        return dict(
            images=images,
            texts=texts,
            class_names=class_names,
            masks=masks,
            img_paths=img_paths,
            cls=cls,
            targets_res=targets_res
        )



# 创建数据集实例
dataset = LMoEDataset(root_dir='/utils/CTdataset/output')
print(len(dataset))
# # 示例：获取第一个样本
# # box,
# origin, conversation_normal, x, conversation_abnormal, class_name, mask, img_path = dataset[0]
#
# #展示origin, conversation_normal, x, conversation_abnormal, class_name, mask, img_path 维度以及类型
# print('origin:',origin.shape,type(origin))
# print('conversation_normal:',conversation_normal,type(conversation_normal))
# print('x:',x.shape,type(x))
# print('conversation_abnormal:',conversation_abnormal,type(conversation_abnormal))
# print('class_name:',class_name,type(class_name))
# print('mask:',type(mask))
# print('mask:',mask.shape,type(mask))
# print('img_path:',img_path,type(img_path))

"""
origin: torch.Size([3, 224, 224]) <class 'torch.Tensor'>
conversation_normal: [{'from': 'human', 'value': 'This is a photo of grid for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part. Is there any anomaly in the image?'}, {'from': 'gpt', 'value': 'No, there is no anomaly in the image.'}] <class 'list'>
x: torch.Size([3, 224, 224]) <class 'torch.Tensor'>
conversation_abnormal: [{'from': 'human', 'value': 'This is a photo of grid for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part. Is there any anomaly in the image?'}, {'from': 'gpt', 'value': 'Yes, there is an anomaly in the image, at the center of the image.'}] <class 'list'>
class_name: bottle <class 'str'>
mask: torch.Size([1, 224, 224]) <class 'torch.Tensor'>
img_path: /home/snowolf/dataset/bottle/train/good/099.png <class 'str'>
"""


