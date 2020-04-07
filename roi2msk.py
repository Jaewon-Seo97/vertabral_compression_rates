from read_roi import read_roi_file
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os

rfolpath= "C:/Users/user/Untitled Folder/roi"
rfolname= os.listdir(rfolpath)

print(rfolname)

for fl in range(len(rfolname)):

    rpath= "C:/Users/user/Untitled Folder/roi/"+rfolname[fl]+"/*"
    rfname= glob.glob(rpath)

    roiname= os.listdir("C:/Users/user/Untitled Folder/roi/"+rfolname[fl])
    locset = []
    for rf in range(len(rfname)):

        rname, ext= os.path.splitext(roiname[rf])
        rfile= rfname[rf]

        roi= read_roi_file(rfile)



        x= roi[rname]['x']
        y= roi[rname]['y']

        pn= len(x)
        loc= np.ndarray((pn,2), dtype= np.int32)

        for lo in range(len(x)):

            loc[lo]= (x[lo],y[lo])

        locset.append(loc)

    print(locset[0])

    img= np.zeros((2048,2500,3), np.uint8)
    for fw in range(len(locset)):

        img1= cv2.fillPoly(img, [locset[fw]], (255,255,255))


    mskp= "C:/Users/user/Untitled Folder/mask/"
    mskn= "%s.png" %rfolname[fl]
    cv2.imwrite(mskp+mskn, img1)

    plt.figure(figsize=(30,30))
    plt.imshow(img1)

for fl in range(len(rfolname)):

    rpath= "C:/Users/user/Untitled Folder/roi/"+rfolname[fl]+"/*"
    rfname= glob.glob(rpath)

    roiname= os.listdir("C:/Users/user/Untitled Folder/roi/"+rfolname[fl])
    locset = []
    for rf in range(len(rfname)):

        rname, ext= os.path.splitext(roiname[rf])
        rfile= rfname[rf]

        roi= read_roi_file(rfile)

        x= roi[rname]['x']
        y= roi[rname]['y']

        pn= len(x)
        loc= np.ndarray((pn,2), dtype= np.int32)

        for lo in range(len(x)):

            loc[lo]= (x[lo],y[lo])

        locset.append(loc)

    img= np.zeros((2048,2500,3), np.uint8)
    for fw in range(len(locset)):

        img1= cv2.fillPoly(img, [locset[fw]], (255,255,255))

    mskp= "C:/Users/user/Untitled Folder/mask/"
    mskn= "%s.png" %
    cv2.imwrite(mskp+mskn, img1)

    plt.figure(figsize=(30,30))
    plt.imshow(img1)
