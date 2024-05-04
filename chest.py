import os
from skimage import io, transform
import scipy
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
from dataloader import Rescale_akshit
from dataloader import ToTensor_akshit
from dataloader import HumanDataset_akshit
from models import UU2NET # full size version 173.6 MB
import cv2
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import cv2
from tqdm import tqdm
# normalize the predicted SOD probability map

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn


def butter_bandpass(_lowcut, _highcut, _fs, order=5):
    _nyq = 0.5 * _fs
    _low = _lowcut / _nyq
    _high = _highcut / _nyq
    # noinspection PyTupleAssignmentBalance
    _b, _a = scipy.signal.butter(order, [_low, _high], btype='band', output='ba')
    return _b, _a

def butter_bandpass_filter(_data, _lowcut, _highcut, _fs, order=5):
    _b, _a = butter_bandpass(_lowcut, _highcut, _fs, order=order)
    _y = scipy.signal.lfilter(_b, _a, _data)
    return _y


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
def normalize_to_range(lst):
    min_val = min(lst) 
    max_val = max(lst)
    normalized = [(x - min(lst)) / (max(lst) - min(lst)) for x in lst]
    return normalized

# Example usage:
# my_list = [2, 5, 8, 11, 14]
# normalized_list = normalize_to_range(my_list, 0, 1)
# print(normalized_list)


def main():

    # --------- 1. get image path and name ---------
    # model_name='u2net'#u2netp


    # --------- 2. dataloader ---------
    #1. dataloader
    device = "cpu"
    model_dir = "body segmentation weights/uu2net_bce_itr_33000_train_0.038824_1.280589.pth"
    dataset_dir = "case"
    mask_dir='final_json'
    prediction_dir = "predict"
    os.makedirs(prediction_dir, exist_ok=True)
    test_salobj_dataset = HumanDataset_akshit(
        img_path = dataset_dir ,
        # mask_path = dataset_dir ,
        transforms = transforms.Compose([
            Rescale_akshit(300),
            ToTensor_akshit(flag = 1)
        ]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)
    # --------- 3. model define ---------
    img_name_list  = sorted(os.listdir(dataset_dir))
    # print("...load U2NET---173.6 MB")
    net = UU2NET()
    

    if device == "cuda":
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()
    Area_list=[]
    prev_area = 6000

    # --------- 4. inference for each image ---------
    for i_test, data_test in tqdm(enumerate(test_salobj_dataloader),desc='infer',unit="image"):
        # print('testing')

        # print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

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
        mask_visualize = np.zeros((predict_np.shape[0],  predict_np.shape[1],3))
        mask_visualize_2 = np.zeros((predict_np.shape[0],  predict_np.shape[1],3))

        colors_bgr = [
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
        # print(np.unique(predict_np))
        # fig, ax = plt.subplots()
        for i in range(8):
            mask_visualize_2[predict_np == i] =colors_bgr[i]
        for i in range(4,5):
            mask_visualize[predict_np == i] =np.array((255,255,255),np.uint8)
        ig=cv2.imread(os.path.join(dataset_dir,img_name_list[i_test]))
        grey=cv2.cvtColor(mask_visualize.astype(np.uint8),cv2.COLOR_BGR2GRAY)
        grey_rescaled=cv2.resize(grey,(480,540),interpolation=cv2.INTER_LINEAR)
        mask_visualize_2 = cv2.resize(mask_visualize_2,(480,540),interpolation=cv2.INTER_LINEAR)
        grey_rescaled=cv2.GaussianBlur(grey_rescaled,(3,3),0)
        grey_rescaled = cv2.morphologyEx(grey_rescaled, cv2.MORPH_OPEN, kernel=(5,5))
        canny=cv2.Canny(grey_rescaled,250,255)
        Area_list.append(cv2.countNonZero(grey))
        cv2.imshow("Mask",grey)
        cv2.waitKey(1)
        print(cv2.countNonZero(grey))
        # approx_c = []
        # contours,_=cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # # for contour in contours:
        # #     epsilon = 0.01 * cv2.arcLength(contour, True)  # Adjust epsilon value as needed
        # #     approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        # #     approx_c.append(approx_contour)
        # contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
        # print(grey)
        # Area=cv2.contourArea(contours_sorted[0])
        # if Area<1000:
        #     Area = prev_area
        #     Area_list.append(Area)
        #     zero=np.zeros((540,480),np.uint8)
        #     cv2.drawContours(zero,[contours_sorted[0]],-1, (255, 255, 255), 2)
        #     cv2.imshow('grey_rescaled',zero)
        #     cv2.imshow('image_ori',grey)
        #     key=cv2.waitKey(0)
        #     if key==ord('q'):
        #         continue
        #     print("bad")
        # perv_contour=contours_sorted[0]
        # Area_list.append(Area)
        # # if abs(Area - prev_area) > 2500:
        # #     Area = prev_area
        # # print(Area)
        # # for contour in contours:
        # moment=cv2.moments(contours_sorted[0])
        # centroid_x = int(moment["m10"] / moment["m00"])
        # centroid_y = int(moment["m01"] / moment["m00"])
        # contour_image=cv2.drawContours(mask_visualize_2,contours,-1,(255,255,255),3)
        #final_image=cv2.polylines(mask_visualize_2,contours,True,(255,255,255),3)
        # cv2.imshow('image_ori',mask_visualize_2)
        # cv2.imshow('mask',mask_visualize)
        # print(len(contours))
        # print(contours)
        #img_rescaled=cv2.resize(ig,(480,540),interpolation=cv2.INTER_LINEAR)
        #cv2.circle(img_rescaled,(centroid_x,centroid_y),4,(255,255,255),4)
        # cv2.circle(final_image,(centroid_x*300,centroid_y*300),100,(255,255,255),100)
        # cv2.imshow('reconstructed',final_image)
        
        # cv2.imshow('original image',img_rescaled)
        # # cv2.imshow('contours',contour_image)
        # cv2.imshow("image",canny)
        # # print(final_image.shape, canny.shape, img_rescaled.shape)
        # key=cv2.waitKey(1)
        # if key==ord('q'):
        #     continue
        # mask_visualize = mask_visualize.astype(np.uint8)
        # mask
        # im = ax.imshow(mask_visualize)
        # fig.colorbar(im)
        # plt.show()
        # plt.waitforbuttonpress("q")
        # save results to test_results folder
        # if not os.path.exists(prediction_dir):
        #     os.makedirs(prediction_dir, exist_ok=True)
        # save_output(dataset_dir, img_name_list[i_test],pred,prediction_dir)
        
        # print('dones')
        #prev_area = Area
        del d0
    plt.figure(figsize=(10, 5)) 
    plt.subplot(1, 2, 1)  
    plt.plot(Area_list)
    n_Area = normalize_to_range(Area_list)
    n_Area_smooth = butter_bandpass_filter(n_Area, 0.4, 2, 30, order=3)
    plt.subplot(1, 2, 2)
    #plt.plot(n_Area)
    plt.plot(n_Area_smooth)
    plt.show()

if __name__ == "__main__":
    main()
