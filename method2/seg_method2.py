#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import glob
import cv2
import math
import numpy as np
import matplotlib.pylab as plt
import shutil
import pandas as pd

all_try=sorted(glob.glob('./try/*'))

#####모두조정
above30=[]
comp_rates=[]
for at in range(len(all_try)):
    img = cv2.imread(all_try[at],1)
    #     print(img.shape, img.min(), img.max())

    img[img>=127] = 255
    img[img<127] = 0

    gray1 = img[:,:,0]
    thresh = gray1

    contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image=gray1, connectivity=4)

    int_cent = centroids.astype(np.int32)

    if nlabels!= 4:

    #     for i in range(nlabels):
    #         cv2.putText(img=img, text=str(i), org=(int_cent[i][0], int_cent[i][1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255, 0, 0), thickness=2)


        plt.figure(figsize=(40,40))
        plt.imshow(img)

    r_approx = []
    for cnt in contours: 
        
        ep=0.05
        epsilon = ep*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

#         print('기존', ep)
#         print(approx, len(approx))

        i=0
        a=0
        while len(approx) !=4:
            i+=1
            if len(approx)>4:
                if i==20:
                    print(all_try[at])
                    break
                elif i==10:
                    a=0.01
#                     print('jump')
                    continue
                else:
#                     ep+=0.01
                    ep=0.5*np.log(a+1)+0.1
                    epsilon = ep*cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    a+=0.008
#                     print('조정+', ep)
                    continue

            elif len(approx)<4:
                if i==20:
#                     print(all_try[at])
                    break
                elif i==10:
                    a=0.01
#                     print('jump')
                    continue
                else:
#                     ep= ep-0.01
#                     ep=np.exp(-a)-0.9
                    ep=math.pow(10,-a)-0.9
                    epsilon = ep*cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
#                     print('조정-', ep)
                    a+=0.008
#                     a=a-0.01
                    continue
            else:
#                 print('approx=4')
                pass
        
        cv2.drawContours(img, [approx], 0,(255,0,0), 1)           
        r_approx.insert(0,approx)
#         print(len(approx), approx)
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
        print(all_try[at], 'right:', len(right), 'left:', len(left))
        shutil.move(all_try[at], './wrong_corner')
    
    oleft = []
    oright = []

    for w in range(0,6,2):
       

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


    if len(oleft)!=6 & len(oright)!= 6:
        print(all_try[at])

    s = 0
    AVH_ls = []

    pr2 = []

    for k in range(nlabels-1):

        AVH = math.sqrt((left[s][0]-left[s+1][0])**2 + (left[s][1]-left[s+1][1])**2)

        AVH_ls.append(AVH)

        s += 2

    for m2 in range(len(AVH_ls)-2):
        V12= AVH_ls[m2]
        V22= AVH_ls[m2+1]
        V32= AVH_ls[m2+2]
        method2 = (((V12+V32)/2-V22)/((V12+V32)/2))*100
        comp_rates.append(method2)
        over=all_try[at]+','+str(method2)
        if method2 >=30.0:
            above30.append(over)
#             plt.figure(figsize=(10,10))
#             plt.imshow(img, cmap='gray')
#             print('%s pressure rate: %.3f' %(all_try[at], method2) + '%')
#             plt.show()

            
#         print('%s pressure rate: %.3f' %(all_try[at], method2) + '%')
        
# print('-----over 30%------')
# print(above30)
# print('\n')


print(len(above30))

# print('done')




data= pd.DataFrame(comp_rates)

data.to_csv('./seg_method2.csv', index=False, header=False)

