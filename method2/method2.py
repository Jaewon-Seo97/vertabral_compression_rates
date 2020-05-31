#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import cv2
import math
import numpy as np
import matplotlib.pylab as plt
import shutil


# In[2]:


bone3files=sorted(glob.glob('C:/Users/user/Untitled Folder/try/msk512/*'))
print(len(bone3files))
print(bone3files[0])


# In[ ]:


no_corner4=sorted(glob.glob('C:/Users/user/Untitled Folder/try/no_corner4/*'))
print(no_corner4)
print(len(no_corner4))


# In[12]:


############한장씩 확인용

img = cv2.imread('C:/Users/user/Untitled Folder/try/msk512/I0000003159_6.jpg',1)
#     print(img.shape, img.min(), img.max())

img[img>=127] = 255
img[img<127] = 0

gray1 = img[:,:,0]
thresh = gray1

# gray = np.float32(gray1)
# img1 = img.copy()

# ret, thresh = cv2.threshold(gray1, 254, 255,0)

contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image=gray1, connectivity=4)

int_cent = centroids.astype(np.int32)

# if nlabels!= 4:

# #     for i in range(nlabels):
# #         cv2.putText(img=img, text=str(i), org=(int_cent[i][0], int_cent[i][1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255, 0, 0), thickness=2)


#     plt.figure(figsize=(40,40))
#     plt.imshow(img)

r_approx = []
for cnt in contours: 


    epsilon = 0.07*cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(img, [approx], 0,(255,0,0), 1)

    r_approx.insert(0,approx)

if len(r_approx[0])& len(r_approx[1])& len(r_approx[2]) !=4:
    print(no_corner4[no4])       
    print(len(r_approx[0]), len(r_approx[1]), len(r_approx[2]))


for i in range(nlabels):
    cv2.putText(img=img, text=str(i), org=(int_cent[i][0], int_cent[i][1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255, 0, 0), thickness=2)
for i in range(0,3):
    for j in range(0,3):
        cv2.putText(img=img, text=str(i), org=(r_approx[i][j][0][0], r_approx[i][j][0][1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255, 0, 0), thickness=2)

plt.figure(figsize=(10,10))    
plt.imshow(img, cmap='gray')
print(left, len(left))
print(right, len(right))


# In[ ]:


########no_corner4, epsilon따로 조절
for no4 in range(len(no_corner4)):
    img = cv2.imread(no_corner4[no4],1)
    #     print(img.shape, img.min(), img.max())

    img[img>=127] = 255
    img[img<127] = 0

    gray1 = img[:,:,0]
    thresh = gray1

    # gray = np.float32(gray1)
    # img1 = img.copy()

    # ret, thresh = cv2.threshold(gray1, 254, 255,0)

    contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image=gray1, connectivity=4)

    int_cent = centroids.astype(np.int32)

    # if nlabels!= 4:

    # #     for i in range(nlabels):
    # #         cv2.putText(img=img, text=str(i), org=(int_cent[i][0], int_cent[i][1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255, 0, 0), thickness=2)


    #     plt.figure(figsize=(40,40))
    #     plt.imshow(img)

    r_approx = []
    for cnt in contours: 


        epsilon = 0.06*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(img, [approx], 0,(255,0,0), 1)

        r_approx.insert(0,approx)

    if len(r_approx[0])& len(r_approx[1])& len(r_approx[2]) !=4:
        print(no_corner4[no4])       
        print(len(r_approx[0]), len(r_approx[1]), len(r_approx[2]))
        
        
    for i in range(nlabels):
        cv2.putText(img=img, text=str(i), org=(int_cent[i][0], int_cent[i][1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255, 0, 0), thickness=2)
    for i in range(0,3):
        for j in range(0,3):
            cv2.putText(img=img, text=str(i), org=(r_approx[i][j][0][0], r_approx[i][j][0][1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255, 0, 0), thickness=2)

    plt.figure(figsize=(10,10))    
    plt.imshow(img, cmap='gray')


#     left=[]
#     right = []
#     for c in range(1,nlabels):
#         if c == nlabels - 1 :

#             x1= centroids[c-1][0]
#             x2= centroids[c][0]
#             y1= centroids[c-1][1]
#             y2= centroids[c][1]


#             for ax in range(0,4):
#                 m = (x2 - x1)/(y2 - y1)
#                 n = x1 - (m*y1)
#                 y = r_approx[c-1][ax][0][1]
#                 x = m*y + n

#                 if r_approx[c-1][ax][0][0] < x:
#                     left.append(r_approx[c-1][ax][0])

#                 else:
#                     right.append(r_approx[c-1][ax][0])

#         else:

#             x1= centroids[c][0]
#             x2= centroids[c+1][0]
#             y1= centroids[c][1]
#             y2= centroids[c+1][1]


#             for ax in range(0,4):
#                 m = (x2 - x1)/(y2 - y1)
#                 n = x1 - (m*y1)
#                 y = r_approx[c-1][ax][0][1]
#                 x = m*y + n

#                 if r_approx[c-1][ax][0][0] < x:
#                     left.append(r_approx[c-1][ax][0])

#                 else:
#                     right.append(r_approx[c-1][ax][0])

#     print(right, len(right))
#     print(left, len(left))


# In[ ]:


print(r_approx)
r_approx[2][1][0][1]


# In[25]:


########approx 따기
above30= []
above20= []
comp_rates=[]
for b3msk in range(len(bone3files)):
    img = cv2.imread(bone3files[b3msk],1)
#     print(img.shape, img.min(), img.max())

    img[img>=127] = 255
    img[img<127] = 0

    gray1 = img[:,:,0]
    thresh = gray1

    # gray = np.float32(gray1)
    # img1 = img.copy()

    # ret, thresh = cv2.threshold(gray1, 254, 255,0)

    contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image=gray1, connectivity=4)

    int_cent = centroids.astype(np.int32)

## bone3 아닌 것 거르기    
#     if nlabels!= 4:
#         print(bone3files[b3msk])
#         shutil.move(bone3files[b3msk], 'C:/Users/user/Untitled Folder/try/no_bone3' )    
        

    r_approx = []
    for cnt in contours: 

        cv2.drawContours(img, [cnt], 0,(255,0,0), 1)
        epsilon = 0.07*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        r_approx.insert(0,approx)

        
####bone3 아니라고 인식한 것 거르기        
#     if len(r_approx)!=3:
#         print(bone3files[b3msk])
#         print(len(r_approx))
#         print(r_approx)
#         shutil.move(bone3files[b3msk], 'C:/Users/user/Untitled Folder/try/no_bone3' )

#         plt.figure(figsize=(10,10))    
#         plt.imshow(img, cmap='gray')




    if len(r_approx[0])& len(r_approx[1])& len(r_approx[2]) !=4:
        print(bone3files[b3msk])       
        print(len(r_approx[0]), len(r_approx[1]), len(r_approx[2]))
        
#         shutil.move(bone3files[b3msk], 'C:/Users/user/Untitled Folder/try/no_corner4' )
#     for i in range(nlabels):
#         cv2.putText(img=img, text=str(i), org=(int_cent[i][0], int_cent[i][1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255, 0, 0), thickness=2)

#     plt.figure(figsize=(10,10))    
#     plt.imshow(img, cmap='gray')

    left=[]
    right = []
    for c in range(1,nlabels):
        if c == nlabels - 1 :

            x1= centroids[c-1][0]
            x2= centroids[c][0]
            y1= centroids[c-1][1]
            y2= centroids[c][1]


            for ax in range(0,4):
                m = (x2 - x1)/(y2 - y1)
                n = x1 - (m*y1)
                y = r_approx[c-1][ax][0][1]
                x = m*y + n

                if r_approx[c-1][ax][0][0] < x:
                    left.append(r_approx[c-1][ax][0])

                else:
                    right.append(r_approx[c-1][ax][0])

        else:

            x1= centroids[c][0]
            x2= centroids[c+1][0]
            y1= centroids[c][1]
            y2= centroids[c+1][1]


            for ax in range(0,4):
                m = (x2 - x1)/(y2 - y1)
                n = x1 - (m*y1)
                y = r_approx[c-1][ax][0][1]
                x = m*y + n

                if r_approx[c-1][ax][0][0] < x:
                    left.append(r_approx[c-1][ax][0])

                else:
                    right.append(r_approx[c-1][ax][0])
    #####corner로 안잡은 것들 거르기    
    if len(right)!= 6 & len(left)!= 6:
        print(bone3files[b3msk], 'right:', len(right), 'left:', len(left))
        shutil.move(bone3files[b3msk], 'C:/Users/user/Untitled Folder/try/wrong_corner')
    oleft = []
    oright = []
    w = 0
    for ul in range(nlabels-1):
        

        if left[w][1] < left[w+1][1]:
            oleft.append(left[w])
            oleft.append(left[w+1])
        else:
            oleft.append(left[w+1])
            oleft.append(left[w])


        if right[w][1] < right[w+1][1]:
            oright.append(right[w])
            oright.append(right[w+1])
        else:
            oright.append(right[w+1])
            oright.append(right[w])

        w += 2


    if len(oleft)!=6 & len(oright)!= 6:
        print(bone3files[b3msk])

    s = 0
    AVH_ls = []

    pr2 = []

    for k in range(nlabels-1):

        AVH = math.sqrt((left[s][0]-left[s+1][0])**2 + (left[s][1]-left[s+1][1])**2)
    #     PVH = math.sqrt((right[s][0]-right[s+1][0])**2 + (right[s][1]-right[s+1][1])**2)

        AVH_ls.append(AVH)
    #     PVH_ls.append(PVH)
        s += 2

    for m2 in range(len(AVH_ls)-2):
        V12= AVH_ls[m2]
        V22= AVH_ls[m2+1]
        V32= AVH_ls[m2+2]
        method2 = (((V12+V32)/2-V22)/((V12+V32)/2))*100
        comp_rates.append(method2)
        over=bone3files[b3msk]+','+str(method2)
        if method2 >=30.0:
            above30.append(over)
        if method2 >=20.0 :
            above20.append(over)
        
            
#         print('%s pressure rate: %.3f' %(bone3files[b3msk], method2) + '%')
        
# print('-----over 30%------')
# print(above30)
# print('\n')

# print('-----over 20%------')
# print(above20)
# print('\n')

# print(len(above30))
# print(len(above20))
print('done')


# In[26]:


import pandas as pd


# In[28]:


len(comp_rates)


# In[34]:


data= pd.DataFrame(comp_rates)

# dat
data.to_csv('C:/Users/user/Untitled Folder/try/npy/regression/method2.csv', index=False, header=False)


# In[32]:


help(data.to_csv)


# In[17]:


type(method2)


# In[ ]:


print(len(oleft))
print(len(oright))

