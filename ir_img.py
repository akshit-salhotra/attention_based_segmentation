import skimage.io as io
import skimage.color as color
import numpy as np
import cv2
img=io.imread('46.jpg')
grey=color.rgb2gray(img)
# print(img)
# print(img.shape)
print(np.unique(np.array(grey.reshape(-1)*255,np.uint8),axis=0).shape)
# print(grey.shape)
data=np.concatenate((grey.reshape(1080,1920,1),grey.reshape(1080,1920,1)),axis=2)

print(data.shape)
io.imshow(img)
io.show()
io.imshow(grey)
io.show()
cv2.imshow('grey',grey)
# key=cv2.waitKey(0)
# if key==ord('q'):
#     cv2.destroyAllWindows()