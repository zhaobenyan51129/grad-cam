import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
from vit_model import vit_base_patch16_224
import json
with open("/home/zhaobenyan/artificial_robustness/imagenet_class_index.json") as f:
    imagenet_classes = {int(i):x[1] for i, x in json.load(f).items()}


class ReshapeTransform:
    def __init__(self, model):
        input_size = model.patch_embed.img_size
        patch_size = model.patch_embed.patch_size
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def __call__(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        result = x[:, 1:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result


def main():
    model = vit_base_patch16_224()
   
    # 链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    weights_path = "./vit_base_patch16_224.pth"
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    # Since the final classification is done on the class token computed in the last attention block,
    # the output will not be affected by the 14x14 channels in the last layer.
    # The gradient of the output with respect to them, will be 0!
    # We should chose any layer before the final attention block.
    '''
    target_layers = [model.pos_drop]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    '''
    images = 'noised'#noised
    if images=='original':
        # load image
        batch_file = '/home/zhaobenyan/artificial_robustness/images_labels_try.pth'
        images_labels = torch.load(batch_file)
        
        #printimages = images_labels['images']('images.shape:',images.shape)
        labels = images_labels['labels']
    else:
        file='/home/zhaobenyan/artificial_robustness/noised_images_tensor.pth'
        label_file='/home/zhaobenyan/artificial_robustness/images_labels_try.pth'
        images_labels = torch.load(label_file)
        labels = images_labels['labels']
        images = torch.load(file)

    img=images.numpy().transpose(0,2,3,1)
    input_tensor=images
    target_category =labels  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog 341

    layers= [[model.pos_drop], [model.blocks[0].mlp], [model.blocks[1].mlp],[model.blocks[2].mlp],
    [model.blocks[3].mlp],[model.blocks[4].mlp],[model.blocks[5].mlp],[model.blocks[6].mlp],
    [model.blocks[7].mlp],[model.blocks[8].mlp] ,[model.blocks[9].mlp],[model.blocks[10].mlp],
    [model.blocks[-1].norm1] ]
    #layers=[[model.blocks[-1].norm1]]
    
    mode = 'single'#single、merged
    for target_layers in layers: 
        cam = GradCAM(model=model,
                    target_layers=target_layers,
                    use_cuda=False,
                    reshape_transform=ReshapeTransform(model))
                    #ReshapeTransform(model)
        
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
        
        if mode =='single':
            for i in range(100):
                grayscale_camm = grayscale_cam[i,:]
                label=imagenet_classes[int(target_category[i])]
                visualization = show_cam_on_image(img[i,:,:,:] , grayscale_camm, use_rgb=True)
                plt.imshow(visualization)
                plt.title('{}'.format(label))
                plt.show()
                dir='/home/zhaobenyan/data/grad_cam/noised_picture{}/'.format(i+1)
                if not os.path.exists(dir):
                    os.mkdir(dir)
                plt.savefig(os.path.join(dir, 'layer{}'.format(layers.index(target_layers))))
                plt.close()  
            print('----------finish single layer{}------------'.format(layers.index(target_layers)))  
        else:
            plt.figure(figsize=(100, 100))
            for i in range(100):
                plt.subplot(10,10,i+1)
                grayscale_camm = grayscale_cam[i,:]
                label=imagenet_classes[int(target_category[i])]
                visualization = show_cam_on_image(img[i,:,:,:] , grayscale_camm, use_rgb=True)
                plt.imshow(visualization)
                plt.title('{}'.format(label))
                plt.xticks([])
                plt.yticks([])
            dir='/home/zhaobenyan/data/grad_cam/merged_noised/'
            if not os.path.exists(dir):
                os.mkdir(dir)
            plt.show()
            plt.savefig(os.path.join(dir, 'layer{}'.format(layers.index(target_layers))))
            plt.close()
            print('----------finish merged layer{}------------'.format(layers.index(target_layers)))          



if __name__ == '__main__':
    main()
