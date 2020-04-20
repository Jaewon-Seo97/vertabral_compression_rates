from read_roi import read_roi_file
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os
import pydicom


rfolpath= "../data/roi"
afolname= sorted(os.listdir(rfolpath))

rfolname= [roiext for roiext in afolname if roiext != '.ipynb_checkpoints']

dipath= '../data/dcm/*'
diname= sorted(glob.glob(dipath))

# roi to mask
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

        for fw in range(len(locset)):

            img1= cv2.fillPoly(img, [locset[fw]], (255,255,255))

        cv2.imwrite("../data/msk/"+rfolname[fl]+'.png', img1)


# make original image, mask image to 512size
pathmsk= "../data/msk/*.png"
folmsk= sorted(glob.glob(pathmsk))

ori_img = sorted(glob.glob("../jin_data/original/img/*"))

for f5 in range(len(folmsk)):

    mskn = "%s.png" %rfolname[f5]

    mspath=folmsk[f5]
    oimg = cv2.imread(mspath)
    mspath= ori_img[f5]
    dimg = cv2.imread(mspath)

    height, width , channel = oimg.shape
#     print(oimg)
    imhei= height
    imwid= width

    rwid= round(int((imwid/imhei)*512))

#     assert round(int((/imwid)*512)) < 512

    rimg= cv2.resize(oimg, dsize=(rwid,512), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
    bimg= np.zeros((512,512))
    bimg[0:512, 0:rwid] = gray

    dimg= cv2.resize(dimg, dsize=(rwid,512), interpolation=cv2.INTER_AREA)
    dimg = cv2.cvtColor(dimg, cv2.COLOR_BGR2GRAY)
    dgray= np.zeros((512,512))
    dgray[0:512, 0:rwid] = dimg


    cv2.imwrite("../data/msk512dum/"+mskn, bimg)
    cv2.imwrite("../data/ori512dum/"+mskn, dgray)

#divide data

import shutil

divo= sorted(glob.glob('../data/ori512dum/*'))
divm= sorted(glob.glob('../data/msk512dum/*'))


# print(len(divf))

for tes in range(0,40):
    otest= divo[tes]
    shutil.copy(otest, '../data/ori/test' )
    mtest= divm[tes]
    shutil.copy(mtest, '../data/msk/test' )

for tra in range(40,200):
    otra= divo[tra]
    shutil.copy(otra, '../data/ori/train' )
    mtra= divm[tra]
    shutil.copy(mtra, '../data/msk/train' )


# print(len(glob.glob("../data/img200/ori/test/*")))
# print(len(glob.glob("../data/img200/msk/test/*")))
# print(len(glob.glob("../data/img200/ori/train/*")))
# print(len(glob.glob("../data/img200/msk/train/*")))
