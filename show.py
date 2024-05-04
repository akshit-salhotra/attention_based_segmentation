import cv2
import os
path='CelebAMask-HQ/CelebAMask-HQ-mask-anno/05269_neck.png'
img=cv2.imread(path)
cv2.imshow('image',img)
key=cv2.waitKey(0)
os.remove(path)
if key==ord('q'):
    cv2.destroyAllWindows()
    