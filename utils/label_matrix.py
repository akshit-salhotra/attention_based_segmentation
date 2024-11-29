label_class_path_list=[]
# (index) 0 class is background
classes=['hair','skin','mouth','nose','eye_g','l_eye','r_eye','l_brow','r_brow','l_ear','r_ear','u_lip','l_lip','hat','neck','ear_r','neck_l','cloth']
import os 
from tqdm import tqdm
import cv2
import numpy as np
# label_path='DUTS/DUTS-TR/DUTS-TR/gt_aug'
# save_label='DUTS/DUTS-TR/DUTS-TR/new_label'
label_path="CelebAMask-HQ/CelebAMask-HQ-mask-anno"
save_labels='array'
count=0
count_file=1
for i,file in tqdm(enumerate(sorted(os.listdir(label_path))),desc='saving',unit='file'):
    # print(type('{:05d}'.format(count)))
    # print('{:05d}'.format(count))
    # print(count)
    # if i<14 or count==1:
    #     continue
    # print(file)
    if not i:
        continue

    if '{:05d}'.format(count) in file:
        count_file+=1# make changes here
    # if '00003' in file:
    # if '00000' in file:
        label_class_path_list.append(os.path.join(label_path,file))
        continue
    else:
    # elif not len(label_class_path_list):
    #     continue
        if not len(label_class_path_list):
            print(file)
            print(f'skipping {count}')
            continue

        # for path in label_class_path_list:
        #     print(path)
        segmented_gt=np.zeros((512,512),dtype=np.uint8)
        # print(f'uniques before :{np.unique(segmented_gt)}')
        # for label in label_class_path_list:
        #     greyscale=cv2.imread(label,cv2.IMREAD_GRAYSCALE)
        #     for i,Class in enumerate(classes):
        #         if Class in os.path.basename(label):
        #             print(f'checked for {i+1}')
        #             if not np.any(greyscale==255):
        #                 print('fuck!!!!')
        #             segmented_gt[greyscale==255]=i+1
        #             if np.any(segmented_gt==i+1):
        #                 print('working')
        # print(f'the segmented_gt has all zeros {np.all(segmented_gt==0)}')
        index_list=[]
        dict_={}
        for i,Class in enumerate(classes):
            for label in label_class_path_list:
                if Class in os.path.basename(label):
                    dict_[Class]=label
                    index_list.append(classes.index(Class)+1)
                    greyscale=cv2.imread(label,cv2.IMREAD_GRAYSCALE)
                    assert len(np.unique(greyscale))==2 ,f' greyscale :{np.unique(greyscale)}   filename: {label} dict {dict_}'
                    # print(f'unique grey :{np.unique(greyscale)}')
                    copy=segmented_gt.copy()
                    segmented_gt[greyscale==255]=i+1
                    # if label=="CelebAMask-HQ/CelebAMask-HQ-mask-anno/00107_l_ear.png":
                    #     print(i)
                    #     if np.array_equal(copy,segmented_gt):
                    #         print('kat gya')
                    #     else:
                    #         print('nhi')
                    #         print(np.unique(segmented_gt))
                        
        # print(index_list)

        # print(np.unique(segmented_gt))        
        # for i in range(19):
            # print(f'is {i} present {np.any(segmented_gt==i)}')

        bgr_image=np.zeros((512,512,3))
        bgr_image[segmented_gt==1,:]=[255,0,0]
        bgr_image[segmented_gt==2,:]=[0,255,0]
        bgr_image[segmented_gt==3,:]=[0,0,255]
        bgr_image[segmented_gt==4,:]=[0, 255, 255]
        bgr_image[segmented_gt==5,:]=[255, 0, 255]
        bgr_image[segmented_gt==6,:]=[255, 255, 0]
        bgr_image[segmented_gt==7,:]=[255, 165, 0]
        bgr_image[segmented_gt==8,:]=[255, 192, 203]
        bgr_image[segmented_gt==9,:]=[128, 0, 128]
        bgr_image[segmented_gt==10,:]=[0, 128, 128]
        bgr_image[segmented_gt==11,:]=[165, 42, 42]
        bgr_image[segmented_gt==12,:]=[255, 215, 0]
        bgr_image[segmented_gt==13,:]=[192, 192, 192]
        bgr_image[segmented_gt==14,:]=[230, 230, 250]
        bgr_image[segmented_gt==15,:]=[75, 0, 130]
        bgr_image[segmented_gt==16,:]=[64, 224, 208]
        bgr_image[segmented_gt==17,:]=[128, 0, 0]
        bgr_image[segmented_gt==18,:]=[128, 128, 0]
        np.save(os.path.join(save_labels,str(count)+'.npy'),segmented_gt,allow_pickle=False)
        # segmented_gt=segmented_gt*225//18
        # segmented_gt=np.reshape(segmented_gt,(512,512,1))
        # segmented_gt= cv2.applyColorMap(segmented_gt, cv2.COLORMAP_HOT)
        # cv2.namedWindow('final',cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('final',512,512)
        # stacked_image=np.hstack((segmented_gt,bgr_image))
        # cv2.imshow('final',stacked_image)
        # cv2.imshow('final',segmented_gt)
        # print(count)
        # print(np.unique(segmented_gt))
        # assert len(np.unique(segmented_gt))==count_file,f'there is a problem in the formulation of matrix :{dict_}  also count_file is {count_file}  segmented_gt :{np.unique(segmented_gt)} index_list {index_list} '
        label_class_path_list=[]
        label_class_path_list.append(os.path.join(label_path,file))
        count+=1
        # cv2.imshow('final',bgr_image)
        # key=cv2.waitKey()
        # if key==ord('q'):
        #     break
        count_file=2
cv2.destroyAllWindows()