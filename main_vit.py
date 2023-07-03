import os
import numpy as np
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
import json
with open("/home/zhaobenyan/artificial_robustness/imagenet_class_index.json") as f:
    imagenet_classes = {int(i):x[1] for i, x in json.load(f).items()}


class ReshapeTransform:
    def __init__(self, model):

        self.h = 14
        self.w = 14

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
    model = torchvision.models.vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_V1')
    #target_layers=[model.encoder.layers.encoder_layer_1.mlp]#, 
    #target_layers=[model.encoder.layers.encoder_layer_11.ln_1]
    
    # load image
    batch_file = '/home/zhaobenyan/artificial_robustness/images_labels.pth'
    images_labels = torch.load(batch_file)
    images = images_labels['images']
    print('images.shape:',images.shape)
    labels = images_labels['labels']

    # load noised_image
    file='/home/zhaobenyan/artificial_robustness/noised_images_tensor.pth'
    images_noised = torch.load(file)
    print('images_noised.shape:',images_noised.shape)

    img=images.numpy().transpose(0,2,3,1)
    input_tensor=images
    target_category =labels  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog 341

    #layers=[model.encoder.layers.encoder_layer_11.ln_1]

    '''
    layers=[model.encoder.layers.encoder_layer_0.mlp, model.encoder.layers.encoder_layer_1.mlp,
    model.encoder.layers.encoder_layer_2.mlp,model.encoder.layers.encoder_layer_3.mlp,
    model.encoder.layers.encoder_layer_4.mlp,model.encoder.layers.encoder_layer_5.mlp,
    model.encoder.layers.encoder_layer_6.mlp,model.encoder.layers.encoder_layer_7.mlp,
    model.encoder.layers.encoder_layer_8.mlp,model.encoder.layers.encoder_layer_9.mlp,
    model.encoder.layers.encoder_layer_10.mlp,model.encoder.layers.encoder_layer_11.ln_1]
    '''
    
    layers=[model.encoder.layers.encoder_layer_10.mlp]
    for target_layers in layers: 
        cam = GradCAM(model=model,
                    target_layers=target_layers,
                    use_cuda=False,
                    reshape_transform=ReshapeTransform(model))
                    #ReshapeTransform(model)
        
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
        for i in range(1):
            #plt.subplot(10,10,i+1)
            grayscale_camm = grayscale_cam[i,:]
            label=imagenet_classes[int(target_category[i])]
            visualization = show_cam_on_image(img[i,:,:,:] , grayscale_camm, use_rgb=True)
            plt.imshow(visualization)
            plt.title('{}'.format(label))
            plt.show()
            dir='/home/zhaobenyan/data/grad_cam/picture{}/'.format(i+1)
            if not os.path.exists(dir):
                os.mkdir(dir)
            plt.savefig(os.path.join(dir, 'embedding_layer_{}'.format(layers.index(target_layers)+1)))
            plt.close()
        # plt.xticks([])
        # plt.yticks([])
    # plt.show()
    # plt.savefig('/home/zhaobenyan/data/grad_cam/cam_original_layer12/original.jpg')
    # plt.close()
    


if __name__ == '__main__':
    main()
