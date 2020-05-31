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


# In[2]:


pwd


# In[12]:


################################################################################
# Data path
################################################################################
npy_save_path = "C:/Users/user/Untitled Folder/try/npy/"

train_ori_path = 'C:/Users/user/Untitled Folder/regression_data_set/image_train/'
train_msk_path = 'C:/Users/user/Untitled Folder/regression_data_set/mask_train/'
test_ori_path = 'C:/Users/user/Untitled Folder/regression_data_set/image_test/'
test_msk_path = 'C:/Users/user/Untitled Folder/regression_data_set/mask_test/'
################################################################################

################################################################################
# Data list
################################################################################
train_image= sorted(glob.glob(train_ori_path+"/*"))
train_mask= sorted(glob.glob(train_msk_path+"/*"))
test_image= sorted(glob.glob(test_ori_path+"/*"))
test_label= sorted(glob.glob(test_msk_path+"/*"))
################################################################################


################################################################################
# Data 수 체크
################################################################################
print(len(train_image))
print(len(train_mask))
print(len(test_image))
print(len(test_label))
################################################################################


################################################################################
# 이미지 크기
################################################################################
image_size = 512
################################################################################


################################################################################
# 빈행렬 생성 (train data와 test data를 각각 따로 만듬 - 그냥 둘다 합쳐서 하나의 파일로 만들어도 되어서 학습하기 전에 train data와 test data를 나눠도 됨)
# 지금은 Segmentation 이기때문에 원본 이미지와 마스크(ground truth)를 각각 만들어줘야함
# 처음에 마스크를 만들때 원본 이미지와 이름을 같게 해서 이미지끼리 순서가 안섞이게 해야함 - 이거는 제대로 했을거라고 믿을게
################################################################################
train_imgs = np.ndarray((len(train_image), image_size, image_size,1), dtype = np.float32)       #딥리닝 학습데이터는 (이미지수,이미지 너비, 이미지 높이, 채널 수)와 같은 형태로 만듬 - 너비랑 높이는 순서가 헷갈리는데 그냥 웬만하면 너비랑 높이는 같은 크기로
train_labels = np.ndarray((len(train_mask), image_size, image_size,1), dtype = np.float32)      #특수한 경우에만 너비랑 높이를 다르게함
test_imgs = np.ndarray((len(test_image), image_size, image_size,1), dtype = np.float32)         #len(train_image)는 이미지의 갯수
test_labels = np.ndarray((len(test_label), image_size, image_size, 1), dtype = np.float32)      #이미지의 데이터 타입은 float32로 만들어야함
################################################################################


# In[13]:


j = 0


for imgname in train_image:
    			              
    midname = imgname[imgname.index("I"):imgname.rindex(".bmp")]  #파일 이름
    
#     print(midname)
#     print(train_ori_path + midname+'.jpg')
    img_1 = load_img(train_ori_path + midname+'.bmp', grayscale = True) #원본이미지를 grayscale을 적용해서 채널을 1개로 만듬
    label_1 = load_img(train_msk_path + midname+'.bmp', grayscale = True) #마스크이미지를 grayscale을 적용해서 채널을 1개로 만듬
    img_1=img_1.resize((image_size,image_size)) #이미지 크기 조정
    label_1=label_1.resize((image_size,image_size)) #이미지 크기 조정
    
    img_1 = img_to_array(img_1) # 위에서 불러온 이미지들을 배열로 만듬
    label_1 = img_to_array(label_1) # 위에서 불러온 이미지들을 배열로 만듬
    

    train_imgs[j] = img_1 #위에 생성한 빈행렬에 이미지를 한장씩 한장씩 쌓아서 하나의 배열로 만듬 (만약 이미지가 100개가 있으면 100개의 이미지 배열이 있을거고 100개의 배열을 하나의 배열로 합치는 과정)
    train_labels[j] = label_1 
    
    j=j+1
    

print('loading done')
np.save(npy_save_path + '/train_512.npy', train_imgs) # 배열 저장
np.save(npy_save_path + '/label_512.npy', train_labels) # 배열 저장
print('Saving to .npy files done.')


i = 0
midname_list=[]   # test data의 파일 이름 저장하는 빈행렬
for imgname in test_image:
    	              
    midname = imgname[imgname.index("I"):imgname.rindex(".bmp")]   # 파일이름
    midname_list.append(midname)  # 파일 이름 저장
#     print(train_ori_path + midname+'.jpg')
    img_2 = load_img(test_ori_path + midname+'.bmp', grayscale = True) # 아래부터는 위와 동일
    label_2 = load_img(test_msk_path + midname+'.bmp', grayscale = True)
    img_2=img_2.resize((image_size,image_size))
    label_2=label_2.resize((image_size,image_size))
    img_2 = img_to_array(img_2)
    label_2 = img_to_array(label_2)
    
    test_imgs[i] = img_2
    test_labels[i] = label_2
    i=i+1
    

print('loading done')
np.save(npy_save_path + '/test_name_512.npy', midname_list) # test data 파일 이름 행렬 저장
np.save(npy_save_path + '/test_512.npy', test_imgs)
np.save(npy_save_path + '/test_label_512.npy', test_labels)
print('Saving to .npy files done.')


# In[14]:


print(train_imgs.shape)
print(train_labels.shape)
print(test_imgs.shape)
print(test_labels.shape)


# In[15]:


################################################################################
# 저장한 데이터가 제대로 만들어 졌는지 테스트하는 코드
################################################################################
train_image_1 = np.load(npy_save_path + '/train_512.npy')
train_label_1 = np.load(npy_save_path + '/label_512.npy')
test_image_1 = np.load(npy_save_path + '/test_512.npy')
test_label_1 = np.load(npy_save_path + '/test_label_512.npy')


################################################################################
# sam 이라는 변수가 몇번째 이미지인지 나타냄 - 바꿔보면서 제대로 됬는지 확인하기
################################################################################
sam = 17
plt.figure(figsize=(20, 20))
plt.subplot(2,3,1)
plt.imshow(train_image_1[sam,:,:,0],cmap='gray')
plt.axis('off')
plt.subplot(2,3,2)
plt.imshow(train_label_1[sam,:,:,0],cmap='gray')
plt.axis('off')
plt.subplot(2,3,3)
plt.imshow(train_image_1[sam,:,:,0],cmap='gray',alpha=0.5)
plt.imshow(train_label_1[sam,:,:,0],cmap='Reds',alpha=0.5)
plt.axis('off')
plt.subplot(2,3,4)
plt.imshow(test_image_1[sam,:,:,0],cmap='gray')
plt.axis('off')
plt.subplot(2,3,5)
plt.imshow(test_label_1[sam,:,:,0],cmap='gray')
plt.axis('off')
plt.subplot(2,3,6)
plt.imshow(test_image_1[sam,:,:,0],cmap='gray',alpha=0.5)
plt.imshow(test_label_1[sam,:,:,0],cmap='Reds',alpha=0.5)


# In[7]:





# In[ ]:




