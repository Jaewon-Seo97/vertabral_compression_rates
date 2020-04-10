from read_roi import read_roi_file
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os
import pydicom


rfolpath= "../data/roi"
afolname= os.listdir(rfolpath)
afolname.sort()
rfolname= [roiext for roiext in afolname if roiext != '.ipynb_checkpoints']

dipath= '../data/dcm/*'
diname= glob.glob(dipath)
diname.sort()

for fl in range(len(rfolname)):

    rpath= "../data/roi/"+rfolname[fl]+"/*"
    rfname= glob.glob(rpath)

    roiname= os.listdir("../data/roi/"+rfolname[fl])
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
    for rc in range(len(diname)):
        ds= pydicom.dcmread(diname[rc])
        imwid= ds.Columns
        imhei= ds.Rows

        img= np.zeros((imwid,imhei), np.uint8)

        for fw in range(len(locset)):

            img1= cv2.fillPoly(img, [locset[fw]], (255,255,255))

        mskp= "../data/msktest/"
        mskn= "%s.png" %rfolname[fl]
        cv2.imwrite(mskp+mskn, img1)
        oimg= cv2.imread(mskp+mskn)

        rwid= round(int((imwid/imhei)*512)
        rimg= cv2.resize(oimg, dsize=(rwid,512), interpolation=cv2.INTER_AREA)
        gray= cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)

        bimg= np.zeros((512,512))
        bimg[0:512, 0:rwid]= gray
        cv2.imwrite("../data/msktest512/"+mskn, bimg)

#         plt.figure(figsize=(10,10))
#         plt.imshow(img1, cmap='gray')
