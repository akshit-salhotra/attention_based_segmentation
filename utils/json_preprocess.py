import os
import json
import cv2
import numpy as np
dir='json'
points=[]
classes=[]
saved_dir="saved"
colors_list=[(0,0,255),(0,255,0),(255,0,255),(33,100,75),(255,0,0),(255,255,255),(0,255,255)]
reconstruction_colors=[(0,0,0),(0,0,255),(0,255,0),(255,0,255),(33,100,75),(255,0,0),(255,255,255),(0,255,255)]
# colors=[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(100,255,165),(0,0,0),(0,0,0),(0,0,0)]

# for i,file in enumerate(os.listdir(dir)):
def json_process(dir,file,colors=colors_list):
    classes=[]
    with open(os.path.join(dir,file)) as f:
        j=json.load(f)
        img=np.zeros((1080,1920,3))
        # print(len(j['objects']))
        # print(file)
        for polygon in j['objects']:
                points=[]
                for point in polygon['polygon']:
                    points.append([point['x'],point['y']])
                cv2.polylines(img,[np.array(points).astype(np.int32)],True,colors[polygon['classIndex']-1],5)
                cv2.fillPoly(img, [np.array(points).astype(np.int32)], colors[polygon['classIndex']-1])     
                classes.append(polygon['classIndex'])
        # print(classes)
        # cv2.imshow('gt',img)
        # key=cv2.waitKey(0)
        # if key==ord('q'):
        #     cv2.destroyAllWindows()        
        matrix=np.zeros((1080,1920),np.uint8)   
        for i,color in enumerate(colors):
            indice=np.where(np.logical_and(np.logical_and(img[:,:,0]==color[0],img[:,:,1]==color[1]),img[:,:,2]==color[2]))
            matrix[indice]=i+1
        
        # temporary fix to make chest,back and torso same
        matrix[matrix==4]=3
        matrix[matrix==5]=3
        matrix[matrix==6]=4
        matrix[matrix==7]=5
        
        
        
        
        return matrix
    
        # print(indice)
        # file,_=os.path.splitext(file)
        # np.save(os.path.join(saved_dir,file+".npy"),matrix)
        # copy=np.reshape(img,(-1,3))
        # print(np.unique(copy,axis=0))
        # cv2.imshow('image',img)
        # reconstruction=np.load(os.path.join(saved_dir,file+'.npy'))
        # new_image=np.zeros((img.shape[0],img.shape[1],3),np.uint8)
        # for i in range(8):
        #     new_image[reconstruction==i,:]=reconstruction_colors[i]
        # print(np.unique(reconstruction))
        # cv2.imshow('reconstructed',new_image)
        # key=cv2.waitKey(1000)
        # if key==ord('q'):
        #     break
        
if __name__ =='__main__':
    # display_color=[(0,0,0),(0,0,255),(0,255,0),(255,0,255),(33,100,75),(255,0,0),(255,255,255),(0,255,255)]
    # json_dir='final_json'
    # for file in os.listdir(json_dir):
    #     matrix=json_process(json_dir,file)          
    #     img=np.zeros((matrix.shape[0],matrix.shape[1],3),np.uint8)
    #     for i in range(6):
    #         img[matrix==i,:]=display_color[i]
    #     cv2.imshow('image',img)
    #     key=cv2.waitKey(2000)
    #     if key=='q':
    #         break          
    # cv2.destroyAllWindows()
    json_process('input_data_body_ir/json','2380.json')