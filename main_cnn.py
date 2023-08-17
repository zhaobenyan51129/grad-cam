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
    batch_tensors = []
    img_list = []

    for img_path in image_paths:
        assert os.path.exists(img_path), "File '{}' does not exist.".format(img_path)
        img = Image.open(img_path).convert('RGB')
        img = cv2.resize(np.array(img, dtype=np.uint8), (224, 224))
        img_tensor = data_transform(img)

        img_list.append(img)
        batch_tensors.append(img_tensor)

    input_tensor = torch.stack(batch_tensors, dim=0)  # [batch,3,224,224]
    img_np = np.array(img_list)
    img_np = img_np.astype(dtype=np.float32) / 255
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
    visualization, heatmap = show_cam_on_image(img,
                                            grayscale_cam,
                                           use_rgb=True)
    batch_size = img.shape[0]
    num_rows = int(np.ceil(np.sqrt(batch_size)))
    num_cols = int(np.ceil(batch_size / num_rows))
 
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    for i, (ax, image) in enumerate(zip(axes.flatten(), visualization)):
        classes = predicted_classes[i]
        ax.imshow(image)
        ax.axis("off")  
        ax.set_title(f"{classes}")  
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    plt.savefig('./data/output/heatmap111.png')
   

if __name__ == '__main__':
    main()