import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img, decode_predictions
import cv2

# 自定义调整大小转换函数
class Resize(object):
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        # 使用 cv2.resize() 进行调整大小
        resized_img = cv2.resize(img, self.target_size)
        return resized_img
    
# 处理数据
def load_images(image_paths):
    data_transform = transforms.Compose([Resize((224, 224)), 
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # 初始化批次张量
    batch_tensors = []
    img_list = []
    # 遍历图像文件路径列表
    for img_path in image_paths:
        assert os.path.exists(img_path), "File '{}' does not exist.".format(img_path)
        
        # 打开图像，进行预处理
        img = Image.open(img_path).convert('RGB')
        img = cv2.resize(np.array(img, dtype=np.uint8), (224, 224))

        img_tensor = data_transform(img)
        
        # 添加到批次张量列表
        img_list.append(img)
        batch_tensors.append(img_tensor)

    # 将批次张量列表堆叠为一个批次张量
    input_tensor = torch.stack(batch_tensors, dim=0)  # [batch,3,224,224]
    img_np = np.array(img_list)
    return input_tensor, img_np


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        use_cuda = True
    else:
        use_cuda = False
    print(f"use_cuda={use_cuda}")

    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    target_layers = [model.features[-1]]        # 可以添加多个层，会对所有层的cam做平均
  
    # 指定图像文件路径列表
    image_paths = [
        "./data/pictures/both.png",
        "./data/pictures/pug-dog.jpg",
        "./data/pictures/pig.jpeg"
    ]

 
    input_tensor, img = load_images(image_paths)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
    # target_category = 281  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog
    # target_category = [281,254]
    target_category = None  # 默认采用预测最高的id
    predicted_classes, grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)  # [batch,H,W]

    batch = grayscale_cam.shape[0]
    plt.figure(figsize=(10, 5))
    for i in range(batch):
        classes = predicted_classes[i]
        grayscale = grayscale_cam[i, :] # 取第i张图片 [224,224]
        visualization = show_cam_on_image(img[i].astype(dtype=np.float32) / 255.,
                                        grayscale,
                                        use_rgb=True)
        plt.subplot(1,batch,i+1)
        plt.imshow(visualization)
        plt.title(f"{classes}")
        # plt.imshow(visualization)
    plt.savefig('./data/output/multiple.png')
    plt.close()

if __name__ == '__main__':
    main()