import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim
import numpy as np
from PIL import Image
import glob
from dataloader_attention import Rescale_withoutmask
from dataloader_attention import ToTensor_withoutmask
from dataloader_attention import HumanDataset_IR_withoutmask
from models import UU2NET # full size version 173.6 MB
from attention_model import U2NETP
from attention_internal_block import Attention_U2NETP
import cv2
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def main():

    # --------- 1. get image path and name ---------
    # model_name='u2net'#u2netp


    # --------- 2. dataloader ---------
    #1. dataloader
    device = "cpu"
    model_dir="best_weights/weights_attention(external +internal) mid dim of internal lesser/attention_bce_itr_20200_train_0.000162_0.013358.pth"
    rgb_dir = "data/test_attention/rgb"
    ir_dir="data/test_attention/ir"
    mask_dir='final_json'
    resize=300
    # prediction_dir = "predict"
    # os.makedirs(prediction_dir, exist_ok=True)
    test_salobj_dataset = HumanDataset_IR_withoutmask(
        img_path = rgb_dir ,
        # mask_path = dataset_dir ,
        ir_path=ir_dir,
        transforms = transforms.Compose([
            Rescale_withoutmask(resize),
            ToTensor_withoutmask(flag = 0)
        ]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    img_name_list  = sorted(os.listdir(rgb_dir))
    
    # print("...load U2NET---173.6 MB")
    # net = UU2NET()
    # net=U2NETP(out_ch=6)
    net=Attention_U2NETP(out_ch=6,attention_weights_flag=1)

    if device == "cuda":
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test= data_test['image']
        # label = data_test['mask']
        # print(inputs_test.size, torch.unique(label))
        inputs_test = inputs_test.type(torch.FloatTensor)
        # gray_test = gray_test.type(torch.FloatTensor)
        if device == "cuda":
            inputs_test = Variable(inputs_test.cuda())
            # gray_test = Variable(gray_test.cuda())
        else:
            inputs_test = Variable(inputs_test)
            # gray_test = Variable(gray_test)

        d0,a1,a2,a3,a4,a5= net(inputs_test)
        a1=np.squeeze(a1.cpu().data.numpy())
        a2=cv2.resize(np.squeeze(a2.cpu().data.numpy()),(resize,resize),interpolation=cv2.INTER_CUBIC)
        a3=cv2.resize(np.squeeze(a3.cpu().data.numpy()),(resize,resize),interpolation=cv2.INTER_CUBIC)
        a4=cv2.resize(np.squeeze(a4.cpu().data.numpy()),(resize,resize),interpolation=cv2.INTER_CUBIC)
        a5=cv2.resize(np.squeeze(a5.cpu().data.numpy()),(resize,resize),interpolation=cv2.INTER_CUBIC)
        
        a1=(((a1-np.min(a1))/(np.min(a1)-np.max(a1)))*255).astype(np.uint8)
        a2=(((a2-np.min(a2))/(np.min(a2)-np.max(a2)))*255).astype(np.uint8)
        a3=(((a3-np.min(a3))/(np.min(a3)-np.max(a3)))*255).astype(np.uint8)
        a4=(((a4-np.min(a4))/(np.min(a4)-np.max(a4)))*255).astype(np.uint8)
        a5=(((a5-np.min(a5))/(np.min(a5)-np.max(a5)))*255).astype(np.uint8)
        
        # print(f'unique values in greyscale {np.unique((((a1-np.min(a1))/(np.min(a1)-np.max(a1))*255).astype(np.uint8)))}')
        # cv2.imshow('greyscale',(((a1-np.min(a1))/(np.min(a1)-np.max(a1)))*255).astype(np.uint8))
        # a1= cv2.applyColorMap((((a1-np.min(a1))/np.max(a1))*255).astype(np.uint8), cv2.COLORMAP_HOT)
        # # print(f'the values of heat map {a1}')
        # a2= cv2.applyColorMap((a1*255).astype(np.uint8), cv2.COLORMAP_HOT)
        # a3= cv2.applyColorMap((a3*255).astype(np.uint8), cv2.COLORMAP_HOT)
        # a4= cv2.applyColorMap((a4*255).astype(np.uint8), cv2.COLORMAP_HOT)
        # a5= cv2.applyColorMap((a5*255).astype(np.uint8), cv2.COLORMAP_HOT)



        # normalization
        predict_np = (d0.cpu())
        predict_np = predict_np.data.numpy()
        predict_np = predict_np.argmax(axis=1)
        predict_np = predict_np.squeeze()
        # print(predict_np.shape)
        mask_visualize = np.zeros((predict_np.shape[0],  predict_np.shape[1], 3))

        colors_bgr = [
    (0,0,0),
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Cyan
    (128, 0, 0),     # Maroon
    (0, 128, 0),     # Olive
    (0, 0, 128),     # Navy
    (128, 128, 0),   # Dark Yellow
    (128, 0, 128),   # Purple
    (0, 128, 128),   # Teal
    (255, 165, 0),   # Orange
    (165, 42, 42),   # Brown
    (128, 128, 128), # Gray
    (255, 255, 255), # White
    (0, 0, 0),       # Black
    (192, 192, 192), # Silver
    (255, 192, 203)  # Pink
]

        print(f'the classes detected are :{np.unique(predict_np)}')
        # fig, ax = plt.subplots()
        
        for i in range(6):
            mask_visualize[predict_np == i] = colors_bgr[i]
        mask_visualize = mask_visualize.astype(np.uint8)
        # # im = ax.imshow(mask_visualize)
        # plt.subplot(1,2,1)
        # plt.imshow(mask_visualize)
        # print(img_name_list[i_test])
        # ir_img=io.imread(f'{ir_dir}/{img_name_list[i_test]}')
        # # ir = ax.imshow(ir_img)
        # plt.subplot(1,2,2)
        # # fig.colorbar(im)
        # # plt.imshow(ir_img)
        # plt.imshow(ir_img)
        # plt.show()
        ir_img=io.imread(f'{ir_dir}/{img_name_list[i_test]}')
        rgb_img=io.imread(f'{rgb_dir}/{img_name_list[i_test]}')
        ir_img=transform.resize(ir_img,(resize,resize),mode='constant')
        rgb_img=transform.resize(rgb_img,(resize,resize),mode='constant')
        # print(ir_img)
        # print(rgb_img.shape)
        mask_visualize=cv2.cvtColor(mask_visualize,cv2.COLOR_RGB2BGR)
        ir_img = cv2.normalize(ir_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        rgb_img = cv2.normalize(rgb_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        ir_img=cv2.cvtColor(ir_img,cv2.COLOR_RGB2BGR)
        rgb_img=cv2.cvtColor(rgb_img,cv2.COLOR_RGB2BGR)
        stacked=cv2.hconcat([mask_visualize,ir_img,rgb_img])
        heatmap=cv2.hconcat([a1,a2,a3,a4,a5])
        
        cv2.imshow('output',stacked)
        cv2.imshow('attention heat maps',heatmap)
        key=cv2.waitKey(1)
        if key==ord('q'):
            cv2.destroyAllWindows()
            break
        
        del d0

if __name__ == "__main__":
    main()
