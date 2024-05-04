import os 
path='CelebAMask-HQ/CelebAMask-HQ-mask-anno'
for i in range(15):
    path_new=os.path.join(path,str(i))
    for file in os.listdir(path_new):
        os.rename(os.path.join(path_new,file),os.path.join(path,file))
        