import numpy as np
import os

dataset_dir = "/home/uas/Desktop/U-2-Net/DUTS/DUTS-TR/DUTS-TR/new_label"

list = sorted(os.listdir(dataset_dir))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
print(len(list))
# print(list[1])
arr = np.load(dataset_dir+ '/' +list[10000])
print(arr.shape, np.unique(arr))