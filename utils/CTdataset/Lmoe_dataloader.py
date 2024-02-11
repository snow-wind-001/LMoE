import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class lmoe_data(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        初始化函数
        :param root_dir: 数据集的根目录，包含两个子目录，每个子目录表示一个类别。
        :param transform: 图片的转换操作。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # 加载数据
        for label in ["anomal", "normal"]:
            label_dir = os.path.join(root_dir, label)
            for folder in os.listdir(label_dir):
                folder_path = os.path.join(label_dir, folder)
                images = []
                for img_name in sorted(os.listdir(folder_path)):
                    img_path = os.path.join(folder_path, img_name)
                    images.append(img_path)
                if len(images) == 21:
                    self.samples.append((images, label))

    def __len__(self):
        """
        返回数据集中样本的数量。
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        根据索引返回一个样本及其标签。
        """
        img_paths, label = self.samples[idx]
        images = [Image.open(img_path) for img_path in img_paths]
        if self.transform:
            images = [self.transform(image) for image in images]
        images_tensor = torch.stack(images)
        label_tensor = torch.tensor(1 if label == "anomal" else 0)
        return images_tensor, label_tensor
'''
# 数据转换操作
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transform = transforms.Compose([
    # transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.RandomResizedCrop((64, 64)),
    normalize,
])

# 创建数据集实例
dataset = fuse_data(root_dir='/home/snowolf/dataset/fuse_images/train', transform=transform)

# 示例：获取第一个样本
sample, label = dataset[-1]

#打印sample的形状和标签
print(sample.shape, label)
#打印标签的类别
print("label is {}".format(label))
# 示例：获取前4个样本
samples, labels = zip(*[dataset[i] for i in range(8)])
samples = torch.stack(samples)
labels = torch.stack(labels)
batch_size, seq_len, C, H, W = samples.size()
print(batch_size, seq_len, C, H, W)
for i in range(seq_len):
    x = samples[:, i, :, :, :]  # 选取当前序列中的图像
    print(x.shape, labels.shape)
    #将图像拼接在一起，并显示出来
    img = transforms.ToPILImage()(torch.cat([img for img in x], dim=1))

img.show()
'''