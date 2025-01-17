import os 
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from skimage import color, io, transform
from scipy.io import loadmat
import random
import torch
from torchvision import transforms
import copy
import cv2
from utils.json_preprocess import json_process
def crop_and_resize_frame(frame, width=1440, height=810):
    # Calculate starting coordinates for cropping
    x_center = frame.shape[1] // 2  # x-coordinate of the center
    y_center = frame.shape[0] // 2  # y-coordinate of the center
    x = max(0, x_center - width // 2)
    y = max(0, y_center - height // 2)
    cropped_frame = frame[y:y + height, x:x + width]
    resized_frame = cv2.resize(cropped_frame, (1920, 1080), interpolation=cv2.INTER_AREA) 
    return resized_frame
class HumanDataset_IR(Dataset): 
    def __init__(self, img_path,ir_path,mask_path, transforms = None):
        self.paths = {"img":img_path, "mask":mask_path,'ir':ir_path}
        self.img_list = sorted(os.listdir(img_path))
        self.ir_list=sorted(os.listdir(ir_path))
        self.mask_list = sorted(os.listdir(mask_path)) #it is important to use sorted here
        self.transforms = transforms
        assert len(self.img_list) == len(self.mask_list) # WE ARE ASSIGNING ALL THE IMPORTANT VALUES TO MEM
        assert len(self.img_list) != 0
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img = io.imread(os.path.join(self.paths['img'], self.img_list[idx]))
        img=color.rgb2gray(img)
        ir_img=io.imread(os.path.join(self.paths['ir'], self.img_list[idx]))
        ir_img=crop_and_resize_frame(ir_img)
        ir_img=color.rgb2gray(ir_img)
        inputs=np.concatenate((img.reshape(img.shape[0],img.shape[1],1),ir_img.reshape(img.shape[0],img.shape[1],1)),axis=2)
        raw_image = io.imread(os.path.join(self.paths['img'], self.img_list[idx]))
        # mask_3 = loadmat(os.path.join(self.paths['mask'], self.mask_list[idx]))['M']
        # mask_3=np.load(os.path.join(self.paths['mask'], self.mask_list[idx]))
        mask_3=json_process(self.paths['mask'], self.mask_list[idx])
        imidx = np.array([idx])
        if len(mask_3.shape) == 3:
            mask = mask_3[:,:,0]
        elif len(mask_3.shape) == 2:
            mask = mask_3
        # if len(img.shape)==3 and len(mask.shape) == 2:
        #     mask = mask[:, :, np.newaxis]
        # elif len(img.shape) and len(mask.shape) == 2:
        #     img = img[:,:,np.newaxis]
        #     mask  = mask[:,:,np.newaxis]
        sample = {"imidx":imidx, "image": inputs, "mask":mask, "rawimg":raw_image}
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample
    
class Rescale(object):
    def __init__(self, out_size):
        assert isinstance(out_size, (int, tuple))
        self.out_size = out_size
    def __call__(self, sample):
        imidx, image, mask, raw_img = sample['imidx'], sample['image'], sample['mask'], sample['rawimg']
        if isinstance(self.out_size, int):
            image = transform.resize(image, (self.out_size, self.out_size), mode = 'constant')
            raw_img = transform.resize(raw_img, (self.out_size, self.out_size), mode = 'constant')
            mask = transform.resize(mask, (self.out_size, self.out_size), mode = 'constant', order = 0, preserve_range = True) #modes are the same as those in np.pad constant means to extend with a constnat value, preserve_range =True this preserves the range however here we are actually reading in float so it will make no differnce
        else:
            image = transform.resize(image, (self.out_size[0], self.out_size[1]), mode = 'constant')
            mask = transform.resize(mask, (self.out_size[0], self.out_size[1]), mode = 'constant', order = 0, preserve_range = True)
        return {'imidx': imidx, 'image': image, "mask":mask,'rawimg':raw_img}
    
class RandomFlip(object):
    def __call__(self, sample):
        imidx, image, mask, raw_img = sample['imidx'], sample['image'], sample['mask'],sample['rawimg']
        if random.random() >= 0.5:
            image = image[::-1]
            mask = mask[::-1]
        return {'imidx': imidx, 'image': image, "mask":mask, 'rawimg':raw_img}
class ToTensor(object):
	def __init__(self,flag=1):
		self.flag = flag
    #this has a temp. fix for accomodating ir + rgb data if flag =0  , np.zeros initialisation has 2 channels instead of 4

	def __call__(self, sample):

		imidx, image, label, raw_image =sample['imidx'], sample['image'], sample['mask'], sample['rawimg']

		# change the color space
		if self.flag == 2: # with rgb and Lab colors
			tmpImg = np.zeros((image.shape[0],image.shape[1],6))
			tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
			if image.shape[2]==1:
				tmpImgt[:,:,0] = image[:,:,0]
				tmpImgt[:,:,1] = image[:,:,0]
				tmpImgt[:,:,2] = image[:,:,0]
			else:
				tmpImgt = image
			tmpImgtl = color.rgb2lab(tmpImgt)

			# nomalize image to range [0,1]
			tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
			tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
			tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
			tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
			tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
			tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
			tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
			tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
			tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

		elif self.flag == 1: #with Lab color
			tmpImg = np.zeros((image.shape[0],image.shape[1],2))
			if image.shape[2]==1:
				tmpImg[:,:,0] = image[:,:,0]
				tmpImg[:,:,1] = image[:,:,0]
				tmpImg[:,:,2] = image[:,:,0]
			else:
				tmpImg = image

			tmpImg = color.rgb2lab(tmpImg)

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

		else: # with rgb color
			tmpImg = np.zeros((image.shape[0],image.shape[1],2))
			image = image/np.max(image)
			if image.shape[2]==1:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
				# tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
			else:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
				# tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225 
        # print('inside')
        # print(tmpImg.shape)
		tmpImg = tmpImg.transpose((2, 0, 1))
		return {'imidx':torch.from_numpy(imidx.copy()), 'image': torch.from_numpy(tmpImg.copy()), 'mask': torch.from_numpy(label.copy()),  'rawimg':raw_image}

class HumanDataset_IR_withoutmask(Dataset): 
    def __init__(self, img_path,ir_path, transforms = None):
        self.paths = {"img":img_path,'ir':ir_path}
        self.img_list = sorted(os.listdir(img_path))
        self.ir_list=sorted(os.listdir(ir_path))
        # self.mask_list = sorted(os.listdir(mask_path))
        #it is important to use sorted here
        self.transforms = transforms
        # assert len(self.img_list) == len(self.mask_list) 
        # WE ARE ASSIGNING ALL THE IMPORTANT VALUES TO MEM
        assert len(self.img_list) != 0
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img = io.imread(os.path.join(self.paths['img'], self.img_list[idx]))
        img=color.rgb2gray(img)
        ir_img=io.imread(os.path.join(self.paths['ir'], self.img_list[idx]))
        ir_img=crop_and_resize_frame(ir_img)
        ir_img=color.rgb2gray(ir_img)
        inputs=np.concatenate((img.reshape(img.shape[0],img.shape[1],1),ir_img.reshape(img.shape[0],img.shape[1],1)),axis=2)
        raw_image = io.imread(os.path.join(self.paths['img'], self.img_list[idx]))
        # mask_3 = loadmat(os.path.join(self.paths['mask'], self.mask_list[idx]))['M']
        # mask_3=np.load(os.path.join(self.paths['mask'], self.mask_list[idx]))
        # mask_3=json_process(self.paths['mask'], self.mask_list[idx])
        imidx = np.array([idx])
        # if len(mask_3.shape) == 3:
        #     mask = mask_3[:,:,0]
        # elif len(mask_3.shape) == 2:
        #     mask = mask_3
        # if len(img.shape)==3 and len(mask.shape) == 2:
        #     mask = mask[:, :, np.newaxis]
        # elif len(img.shape) and len(mask.shape) == 2:
        #     img = img[:,:,np.newaxis]
        #     mask  = mask[:,:,np.newaxis]
        # print(inputs.shape)
        sample = {"imidx":imidx, "image": inputs, "rawimg":raw_image}
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

class Rescale_withoutmask(object):
    def __init__(self, out_size):
        assert isinstance(out_size, (int, tuple))
        self.out_size = out_size
    def __call__(self, sample):
        imidx, image, raw_img = sample['imidx'], sample['image'], sample['rawimg']
        if isinstance(self.out_size, int):
            image = transform.resize(image, (self.out_size, self.out_size), mode = 'constant')
            raw_img = transform.resize(raw_img, (self.out_size, self.out_size), mode = 'constant')
            # mask = transform.resize(mask, (self.out_size, self.out_size), mode = 'constant', order = 0, preserve_range = True) #modes are the same as those in np.pad constant means to extend with a constnat value, preserve_range =True this preserves the range however here we are actually reading in float so it will make no differnce
        else:
            image = transform.resize(image, (self.out_size[0], self.out_size[1]), mode = 'constant')
        # print(image.shape)
            # mask = transform.resize(mask, (self.out_size[0], self.out_size[1]), mode = 'constant', order = 0, preserve_range = True)
        return {'imidx': imidx, 'image': image,'rawimg':raw_img}

class ToTensor_withoutmask(object):
	def __init__(self,flag=1):
		self.flag = flag

	def __call__(self, sample):

		imidx, image, raw_image =sample['imidx'], sample['image'], sample['rawimg']

		# change the color space
		if self.flag == 2: # with rgb and Lab colors
			tmpImg = np.zeros((image.shape[0],image.shape[1],6))
			tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
			if image.shape[2]==1:
				tmpImgt[:,:,0] = image[:,:,0]
				tmpImgt[:,:,1] = image[:,:,0]
				tmpImgt[:,:,2] = image[:,:,0]
			else:
				tmpImgt = image
			tmpImgtl = color.rgb2lab(tmpImgt)

			# nomalize image to range [0,1]
			tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
			tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
			tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
			tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
			tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
			tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
			tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
			tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
			tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

		elif self.flag == 1: #with Lab color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))
			if image.shape[2]==1:
				tmpImg[:,:,0] = image[:,:,0]
				tmpImg[:,:,1] = image[:,:,0]
				tmpImg[:,:,2] = image[:,:,0]
			else:
				tmpImg = image

			tmpImg = color.rgb2lab(tmpImg)

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

		else: # with rgb color
			tmpImg = np.zeros((image.shape[0],image.shape[1],2))
			image = image/np.max(image)
			if image.shape[2]==1:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
				# tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
                #''' temporary fix for ir and rgb'''
			else:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
				# tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225
        


        # print(tmpImg.shape)
		tmpImg = tmpImg.transpose((2, 0, 1))
        # print(tmpImg.shape)
		return {'imidx':torch.from_numpy(imidx.copy()), 'image': torch.from_numpy(tmpImg.copy()),  'rawimg':raw_image}