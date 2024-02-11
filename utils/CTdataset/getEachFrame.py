# 导入所需要的库
import os
import cv2
import json
import numpy as np
from paddleocr import PaddleOCR
from findColor import get_roi
# import pyautogui
# from screeninfo import get_monitors
import logging
from tqdm import tqdm
logging.disable(logging.DEBUG)
logging.disable(logging.WARNING)
import paddle
from skimage.metrics import structural_similarity as ssim
import numpy as np
import re
paddle.set_device("gpu:0")
# 定义保存图片函数
# image:要保存的图片名字
# addr；图片地址与相片名字的前部分
# num: 相片，名字的后缀 int 类型
def save_image(image, addr, num):
    address = addr + str(num) + '.jpg'
    cv2.imwrite(address, image)
    return address

def image_cut_save(path, left, upper, right, lower, save_path):
    """
        所截区域图片保存
    :param path: 图片路径
    :param left: 区块左上角位置的像素点离图片左边界的距离
    :param upper：区块左上角位置的像素点离图片上边界的距离
    :param right：区块右下角位置的像素点离图片左边界的距离
    :param lower：区块右下角位置的像素点离图片上边界的距离
     故需满足：lower > upper、right > left
    :param save_path: 所截图片保存位置
    """
    img = cv2.imread(path)  # 打开图像
    cropped = img[upper:lower, left:right]

    # 保存截取的图片
    cv2.imwrite(save_path, cropped)

def cut_save(img, left, upper, right, lower, save_path):
    """
        所截区域图片保存
    :param img: 图片
    :param left: 区块左上角位置的像素点离图片左边界的距离
    :param upper：区块左上角位置的像素点离图片上边界的距离
    :param right：区块右下角位置的像素点离图片左边界的距离
    :param lower：区块右下角位置的像素点离图片上边界的距离
     故需满足：lower > upper、right > left
    :param save_path: 所截图片保存位置 """
    cropped = img[upper:lower, left:right]
    # 保存截取的图片
    cv2.imwrite(save_path, cropped)
    return cropped

def cut(img, left, upper, right, lower):
    """
        所截区域图片保存
    :param img: 图片
    :param left: 区块左上角位置的像素点离图片左边界的距离
    :param upper：区块左上角位置的像素点离图片上边界的距离
    :param right：区块右下角位置的像素点离图片左边界的距离
    :param lower：区块右下角位置的像素点离图片上边界的距离
     故需满足：lower > upper、right > left
    :param save_path: 所截图片保存位置 """
    cropped = img[upper:lower, left:right]
    return cropped

# 获取病人ID号，创立json文件 # (1320, 180, 1411, 195)
def getInfo(img):
    ocr = PaddleOCR(use_gpu=True)
    print('the process of getting the ID of the patient')
    img=cut(img, 1250, 180, 1410, 200, )
    result = ocr.ocr(img)

    for word_info in result:
        word = word_info[0][1][0]
        print("\n")
        print(word[3:])
        name = word[3:]
    return name


# 获取检测照片数目
def get_detect_num(image):
    ocr = PaddleOCR(use_gpu=True)
    result = ocr.ocr(image)
    target_characters = ["Im:", "/"]
    # print(result)
    return result


def get_detect_res(img):
    # 设置要检测的特定字符
    target_characters = ["病灶测量"]
    # 初始化PaddleOCR
    ocr = PaddleOCR(use_gpu=True)
    # 进行OCR识别
    result = ocr.ocr(img)
    img_info_1 = []
    infor = []
    # 检查是否有特定字符出现
    for line in result:
        for word_info in line:
            word = word_info[-1]
            for char in target_characters:
                if char in word:
                    img_info_1 = cut(img, 0, int(word_info[0][0][1]-100), 500, int(word_info[0][0][1]))
                    result = ocr.ocr(img_info_1)
                    for i in range(len(result[0]) - 1):
                        # 将result[0][i][1][0]内容存入infor
                        infor.append(result[0][i][1][0])
                    break
                else:
                    continue

    return infor

def extract_number(s):
    # 使用冒号分割字符串，并取第二部分
    part_after_colon = s.split(":")[1].strip()

    # 使用斜杠分割字符串，并取第一部分
    number = part_after_colon.split("/")[0].strip()

    # 转换为整数
    return int(number)

def video_process(video_path,classes):
    video_name = video_path
    # read the mp4 video file
    videoCapture = cv2.VideoCapture(video_name)
    #保存第一帧图片
    success, frame = videoCapture.read()
    if success:
        print("read success!")
    else:
        print("read failed!")

    name = getInfo(frame)
    # 检测图像编号
    result = get_detect_num(cut(frame, 9, 150, 80, 220))
    try:
        frame_num_area = frame[150:220, 9:80]
        num_CT = result[0][2][1][0]  # num_CT='Im:188/597'
        num = extract_number(num_CT)
    except Exception as e:
        print(f"error:{e}")
        num = 0
    videoCapture.release()  # 释放视频文件
    videoCapture = cv2.VideoCapture(video_name)
    filedir = f'./output/{classes}/' + video_path.split('.')[0].split('/')[-1]
    os.makedirs(filedir, exist_ok=True)
    # print(filedir)
    while True:
        success, frame = videoCapture.read()
        if not success:
            break  # 如果没有更多的帧，则退出循环
        frame = np.array(frame)
        # cv2.imshow("frame", frame)
        # 检测图像编号
        frame_num = frame[150:220, 9:80]
        # 计算 SSIM 之前判断图像大小，确定合适的 win_size
        min_dimension = min(frame_num_area.shape[:2])  # 获取图像区域的最小维度
        win_size = min(5, min_dimension - (min_dimension % 2 - 1))  # 确保是小于等于最小维度的奇数

        # 使用显式的 win_size 计算 SSIM
        ssim_index = ssim(frame_num_area, frame_num, win_size=win_size, multichannel=True)
        # 设置阈值
        threshold = 0.99  # 根据需要调整，接近1表示非常相似
        if ssim_index > threshold:
            print("area is same!")
            continue
        else:
            pass
        result = get_detect_num(frame_num)
        if result is None:
            continue
        else:
            if len(result[0]) < 3:
                continue
            else:
                frame_num_area = frame[150:220, 9:80]
                num_CT = result[0][2][1][0]  # num_CT='Im:188/597'
                currentnum = extract_number(num_CT)
                print("the current number of the picture is:{}".format(currentnum))
                if currentnum == num:
                    print("pass")
                    continue
                else:
                    num = currentnum
                    img_save_path = filedir + '/' + str(currentnum) + '.jpg'
                    #判断当前文件夹下是否有该文件
                    if os.path.exists(img_save_path):
                        print("the picture is 纵隔窗！")
                        img_save_path = filedir + '/' + str(currentnum) + '_z' + '.jpg'
                        cropped = cut_save(frame, 2, 157, 1417, 1006, img_save_path)
                    else:
                        cropped = cut_save(frame, 2, 157, 1417, 1006, img_save_path)
                    left, upper, right, lower = 1417, 235, 1919, 695
                    # store the picture of the special location
                    img_info = cut(frame, left, upper, right, lower)
                    try:
                        char = get_detect_res(img_info)
                        border_list = get_roi(cropped)
                    except Exception as e:
                        print(f"error:{e}")
                        continue
                    # 构建dict内容写入json
                    new_data = {'id': name, 'img_path' : img_save_path, 'info': char, 'border': border_list, 'class': classes}
                    # print("write successfully!")
                    filename = f'./output/{classes}/'+name + '.json'
                    with open(filename, "a") as f:
                        json.dump(new_data, f, ensure_ascii=False)
                        f.write('\n')
    videoCapture.release()  # 释放视频文件


if __name__ == "__main__":
    print('start getting!')
    """  
    处理指定目录下的所有mp4文件。  
    """
    number = 19
    directory = '/home/snowolf/colmap_leaf/videoCT/' + str(number)
    classes = directory.split('/')[-1].split('MP4')[0]
    print(classes)
    # 获取指定目录下所有的mp4文件
    mp4_files = [f for f in os.listdir(directory) if f.endswith('.mp4')]
    # 创建tqdm进度条
    pbar = tqdm(total=len(mp4_files), ncols=80)
    for mp4_file in mp4_files:
        file_path = os.path.join(directory, mp4_file)
        try:
            video_process(file_path, classes)
        except Exception as e:
            print(f"error:{e}")
            continue

        # 更新进度条
        pbar.update()
    pbar.close()  # 关闭进度条
    print("所有mp4文件处理完成。")
