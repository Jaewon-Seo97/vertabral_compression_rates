#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import glob
import h5py
import cv2
import numpy as np
import matplotlib.pylab as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# In[9]:


npy_save_path = "C:/Users/user/Untitled Folder/try/npy/regression/"

msk512_path = 'C:/Users/user/Untitled Folder/try/msk512/'


msk512_image= sorted(glob.glob(msk512_path+"/*"))

image_size=512

msk512_imgs = np.ndarray((len(msk512_image), image_size, image_size,1), dtype = np.float32) 


# In[11]:


j=0
for imgname in msk512_image:
    			              
    midname = imgname[imgname.index("I"):imgname.rindex(".jpg")]  #파일 이름
    
#     print(msk512_path + midname+'.jpg')
    img_1 = load_img(msk512_path + midname+'.jpg', grayscale = True) #원본이미지를 grayscale을 적용해서 채널을 1개로 만듬
#     label_1 = load_img(train_msk_path + midname+'.bmp', grayscale = True) #마스크이미지를 grayscale을 적용해서 채널을 1개로 만듬
    img_1=img_1.resize((image_size,image_size)) #이미지 크기 조정
#     label_1=label_1.resize((image_size,image_size)) #이미지 크기 조정
    
    img_1 = img_to_array(img_1) # 위에서 불러온 이미지들을 배열로 만듬
#     label_1 = img_to_array(label_1) # 위에서 불러온 이미지들을 배열로 만듬
    

    msk512_imgs[j] = img_1 #위에 생성한 빈행렬에 이미지를 한장씩 한장씩 쌓아서 하나의 배열로 만듬 (만약 이미지가 100개가 있으면 100개의 이미지 배열이 있을거고 100개의 배열을 하나의 배열로 합치는 과정)
#     train_labels[j] = label_1 
    
    j=j+1
    

print('loading done')
np.save(npy_save_path + '/msk512_3bone.npy', msk512_imgs) # 배열 저장
# np.save(npy_save_path + '/label_512.npy', train_labels) # 배열 저장
print('Saving to .npy files done.')

