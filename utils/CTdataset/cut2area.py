import cv2
import os
from PIL import Image

# 全局变量用于存储框选坐标和图像缩放
crop_rectangle = None
scale = 1.0


def on_mouse(event, x, y, flags, param):
    global crop_rectangle, img, scale
    if event == cv2.EVENT_LBUTTONDOWN:
        crop_rectangle = (int(x / scale), int(y / scale), int(x / scale), int(y / scale))
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        crop_rectangle = (crop_rectangle[0], crop_rectangle[1], int(x / scale), int(y / scale))
        tmp_img = cv2.resize(img, None, fx=scale, fy=scale)
        cv2.rectangle(tmp_img, (int(crop_rectangle[0] * scale), int(crop_rectangle[1] * scale)),
                      (int(crop_rectangle[2] * scale), int(crop_rectangle[3] * scale)), (0, 255, 0), 2)
        cv2.imshow("Image", tmp_img)
    elif event == cv2.EVENT_LBUTTONUP:
        crop_rectangle = (crop_rectangle[0], crop_rectangle[1], int(x / scale), int(y / scale))
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:  # Scroll up
            scale *= 1.1
        else:  # Scroll down
            scale /= 1.1

        scaled_img = cv2.resize(img, None, fx=scale, fy=scale)
        cv2.imshow("Image", scaled_img)


def crop_image(image_path, crop_area):
    with Image.open(image_path) as im:
        cropped_img = im.crop(crop_area)
        cropped_img.save(image_path)
        print(f"Cropped and saved {image_path}")


def crop_images_in_folder(folder_path):
    global crop_rectangle, img, scale
    first_image = True  # 标记是否是第一张图片

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)

            if first_image:
                img = cv2.imread(image_path)
                cv2.imshow('Image', img)
                cv2.setMouseCallback('Image', on_mouse)

                while True:
                    key = cv2.waitKey(1)
                    if key == 13:  # 回车键的键码
                        if crop_rectangle and crop_rectangle[2] - crop_rectangle[0] > 0 and crop_rectangle[3] - \
                                crop_rectangle[1] > 0:
                            crop_image(image_path, crop_rectangle)
                        break
                cv2.destroyAllWindows()
                first_image = False
            else:
                crop_image(image_path, crop_rectangle)


# 指定文件夹
folder_path = '/home/snowolf/colmap_leaf'  # 修改为你的图片文件夹路径
crop_images_in_folder(folder_path)