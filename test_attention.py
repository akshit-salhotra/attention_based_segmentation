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
# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(dataset_dir, image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(dataset_dir+os.sep +image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    # cv2_imo = cv2.cvtColor(imo, cv2.COLOR_RGB2BGR)
    

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')

def main():

    # --------- 1. get image path and name ---------
    # model_name='u2net'#u2netp


    # --------- 2. dataloader ---------
    #1. dataloader
    device = "cuda"
    # model_dir = "weights/attention_bce_itr_12600_train_0.000344_0.018050.pth"
    model_dir="weights/attention_bce_itr_19600_train_0.000165_0.009389.pth"
    # dataset_dir = "rgb"
    dataset_dir = "input_data_body_ir/rgb"
    
    # ir_dir="ir"
    ir_dir="input_data_body_ir/ir"
    mask_dir='final_json'
    prediction_dir = "predict"
    os.makedirs(prediction_dir, exist_ok=True)
    test_salobj_dataset = HumanDataset_IR_withoutmask(
        img_path = dataset_dir ,
        # mask_path = dataset_dir ,
        ir_path=ir_dir,
        transforms = transforms.Compose([
            Rescale_withoutmask(300),
            ToTensor_withoutmask(flag = 0)
        ]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    img_name_list  = sorted(os.listdir(dataset_dir))
    
    # print("...load U2NET---173.6 MB")
    # net = UU2NET()
    # net=U2NETP(out_ch=6)
    net=Attention_U2NETP(out_ch=6)

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

        d0 = net(inputs_test)

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

        print(np.unique(predict_np))
        # fig, ax = plt.subplots()
        
        for i in range(6):
            mask_visualize[predict_np == i] = colors_bgr[i]
        mask_visualize = mask_visualize.astype(np.uint8)
        # im = ax.imshow(mask_visualize)
        plt.subplot(1,2,1)
        plt.imshow(mask_visualize)
        print(img_name_list[i_test])
        ir_img=io.imread(f'{ir_dir}/{img_name_list[i_test]}')
        # ir = ax.imshow(ir_img)
        plt.subplot(1,2,2)
        # fig.colorbar(im)
        # plt.imshow(ir_img)
        plt.imshow(ir_img)
        plt.show()
        # plt.waitforbuttonpress("q")

        # save results to test_results folder
        # if not os.path.exists(prediction_dir):
        #     os.makedirs(prediction_dir, exist_ok=True)
        # save_output(dataset_dir, img_name_list[i_test],pred,prediction_dir)

        del d0

if __name__ == "__main__":
    main()
