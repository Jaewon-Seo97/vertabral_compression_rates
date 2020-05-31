#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import cv2
import math
import numpy as np
import matplotlib.pylab as plt
from read_roi import read_roi_file
from read_roi import read_roi_zip
import pydicom


# In[2]:


dcmpath='C:/Users/user/Untitled Folder/spine/'
mskpath= 'C:/Users/user/Untitled Folder/try/msk/'
msk512path='C:/Users/user/Untitled Folder/try/msk512/'
roizipname= sorted(os.listdir('C:/Users/user/Untitled Folder/roi_zip'))
roipath= sorted(glob.glob('C:/Users/user/Untitled Folder/roi_zip/*'))
pnames=sorted(os.listdir('C:/Users/user/Untitled Folder/roi_file'))


# In[23]:


print(len(roizipname))
print(roizipname[0])


# In[12]:


help(os.listdir)


# In[8]:


print(pnames)


# In[9]:


pnames=[]
for names in range(len(roizipname)):
    pname, ext = os.path.splitext(roizipname[names])
    pnames.append(pname)


# In[11]:


print(pnames)


# In[14]:


roinames= os.listdir('C:/Users/user/Untitled Folder/roi_file/'+pnames[0] )
print(roinames)
for one_ROI in range(len(roinames)):
    rname, ext = os.path.splitext(roinames[one_ROI])
    print(rname)
    


# In[37]:


roiname


# In[13]:


roi= read_roi_zip('C:/Users/user/Untitled Folder/roi_zip/'+roizipname[0])
roiname= os.listdir('C:/Users/user/Untitled Folder/roi_file/'+pnames[0])
# print(roi)
# roizipname[0]
###################### roi
locset = []
for rf in range(len(roiname)):
    rname, ext= os.path.splitext(roiname[rf])
    print(rname)
    print(roi[rname]['type'])
    x= roi[rname]['x']
    y= roi[rname]['y']

    pn= len(x)
    loc= np.ndarray((pn,2), dtype= np.int32)

    for lo in range(len(x)):

        loc[lo]= (x[lo],y[lo])

    locset.append(loc)
print(locset[0])


# In[ ]:





# In[ ]:



for one_p in range(len(pnames)):
    
    roiname= os.listdir('C:/Users/user/Untitled Folder/roi_file/'+pnames[one_p])
    roi= read_roi_zip('C:/Users/user/Untitled Folder/roi_zip/'+roizipname[one_p])
    ###################### roi
    locset = []
    for rf in range(len(roiname)):
        rname, ext= os.path.splitext(roiname[rf])
        
        if roi[rname]['type']!= 'polygon':
            print(pnames[one_p])
            
            pass
        
        else:


            x= roi[rname]['x']
            y= roi[rname]['y']

            pn= len(x)
            loc= np.ndarray((pn,2), dtype= np.int32)

            for lo in range(len(x)):

                loc[lo]= (x[lo],y[lo])

            locset.append(loc)


        #################### mask
        ds= pydicom.dcmread(dcmpath + pnames[one_p]+'.dcm')

        imhei= ds.Columns
        imwid= ds.Rows

        img= np.zeros((imwid,imhei), np.uint8)

        for fw in range(len(locset)-2):
            backimg= img.copy()

            for b3 in range(0,3):
                img1= cv2.fillPoly(backimg, [locset[b3+fw]], (255,255,255))
            aa=str(fw+2)
            print(str(aa))
            cv2.imwrite(mskpath+ pnames[one_p]+ '_' + aa +'.jpg', img1)

        #     plt.figure(figsize=(20,20))
        #     plt.imshow(img1, cmap='gray')



        ####################### 512   

            omimg = cv2.imread(mskpath+ pnames[one_p]+ '_' + aa +'.jpg')
            height, width , channel = omimg.shape

            rwid= round(int((width/height)*512))

            rimg= cv2.resize(omimg, dsize=(rwid,512), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
            bimg= np.zeros((512,512))
            bimg[0:512, 0:rwid] = gray

        #     plt.figure(figsize=(20,20))
        #     plt.imshow(cv2.imread('C:/Users/user/Untitled Folder/regression_data_set/image/I0000007229.bmp'))
        #     plt.imshow(bimg, cmap='gray',alpha=0.5)

        #     plt.show()
            cv2.imwrite(msk512path+ pnames[one_p]+ '_' + aa +'.jpg', bimg)


# In[7]:


#######죽어서 다시 시작
for one_p in range(355,529):
    
    roiname= os.listdir('C:/Users/user/Untitled Folder/roi_file/'+pnames[one_p])
    roi= read_roi_zip('C:/Users/user/Untitled Folder/roi_zip/'+roizipname[one_p])
    ###################### roi
    locset = []
    for rf in range(len(roiname)):
        rname, ext= os.path.splitext(roiname[rf])
        
        if roi[rname]['type']!= 'polygon':
            print(pnames[one_p])
            
            pass
        
        else:


            x= roi[rname]['x']
            y= roi[rname]['y']

            pn= len(x)
            loc= np.ndarray((pn,2), dtype= np.int32)

            for lo in range(len(x)):

                loc[lo]= (x[lo],y[lo])

            locset.append(loc)


        #################### mask
        ds= pydicom.dcmread(dcmpath + pnames[one_p]+'.dcm')

        imhei= ds.Columns
        imwid= ds.Rows

        img= np.zeros((imwid,imhei), np.uint8)

        for fw in range(len(locset)-2):
            backimg= img.copy()

            for b3 in range(0,3):
                img1= cv2.fillPoly(backimg, [locset[b3+fw]], (255,255,255))
            aa=str(fw+2)
#             print(str(aa))
            cv2.imwrite(mskpath+ pnames[one_p]+ '_' + aa +'.jpg', img1)

        #     plt.figure(figsize=(20,20))
        #     plt.imshow(img1, cmap='gray')



        ####################### 512   

            omimg = cv2.imread(mskpath+ pnames[one_p]+ '_' + aa +'.jpg')
            height, width , channel = omimg.shape

            rwid= round(int((width/height)*512))

            rimg= cv2.resize(omimg, dsize=(rwid,512), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
            bimg= np.zeros((512,512))
            bimg[0:512, 0:rwid] = gray

        #     plt.figure(figsize=(20,20))
        #     plt.imshow(cv2.imread('C:/Users/user/Untitled Folder/regression_data_set/image/I0000007229.bmp'))
        #     plt.imshow(bimg, cmap='gray',alpha=0.5)

        #     plt.show()
            cv2.imwrite(msk512path+ pnames[one_p]+ '_' + aa +'.jpg', bimg)
            
print('done')

