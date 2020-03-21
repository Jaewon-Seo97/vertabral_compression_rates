import os
import glob
import h5py
import cv2
import math
import numpy as np
import matplotlib.pylab as plt


path = '../data/data_set/test/msk/*'
filename = glob.glob(path)

for f in range(len(filename)):

    print('-'*10 + 'file%d' %(f+1) + '-'*10)
    file = filename[f]

    img = cv2.imread(file)

    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray1)
    img1 = img.copy()

    ret, thresh = cv2.threshold(gray1,127,255,0)

    contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray1)

    int_cent = centroids.astype(np.int32)



    r_approx = []
    for cnt in contours:

        epsilon = 0.095*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(img1, [approx], 0,(255,0,0), 1)

        r_approx.insert(0,approx)



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

    #                 print('left')
    #                 print('m: %f, y: %f, x: %f, r_approx: %d' %(m, y, x, r_approx[c-1][ax][0][0]))
    #                 print(left)

                else:
                    right.append(r_approx[c-1][ax][0])
    #                 print('right')
    #                 print('m: %f, y: %f, x: %f, r_approx: %d' %(m, y, x, r_approx[c-1][ax][0][0]))
    #                 print(right)
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

    #                 print('left')
    #                 print('m: %f, y: %f, x: %f, r_approx: %d' %(m, y, x, r_approx[c-1][ax][0][0]))
    #                 print(left)

                else:
                    right.append(r_approx[c-1][ax][0])


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


#     for i in range(len(oright)):
#         cv2.putText(img1, str(i), (oright[i][0],oright[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 2)
#     for i in range(len(oleft)):
#         cv2.putText(img1, str(i), (oleft[i][0],oleft[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)


    s = 0
    AVH_ls = []
    PVH_ls = []

    pr1 = []
    pr2 = []
    pr3 = []
    pr4 = []
    for k in range(nlabels-1):

        AVH = math.sqrt((left[s][0]-left[s+1][0])**2 + (left[s][1]-left[s+1][1])**2)
        PVH = math.sqrt((right[s][0]-right[s+1][0])**2 + (right[s][1]-right[s+1][1])**2)

        AVH_ls.append(AVH)
        PVH_ls.append(PVH)
        s += 2

        m1 = (1-(AVH/PVH))*100

        pr1.append(m1)




    print('method 1')
    for p1 in range(len(pr1)):
        print('%d pressure rate: %.3f' %(p1+1, pr1[p1]) + '%')



    print("method2")
    for m2 in range(len(AVH_ls)-2):
        V12= AVH_ls[m2]
        V22= AVH_ls[m2+1]
        V32= AVH_ls[m2+2]
        method2 = (((V12+V32)/2-V22)/((V12+V32)/2))*100


        print('%d pressure rate: %.3f' %(m2+2, method2) + '%')


    print("method3")
    for m3 in range(len(AVH_ls)-1):
        V13= AVH_ls[m3]
        V23= AVH_ls[m3+1]
        method3 = (1-V23/V13)*100

        print('%d pressure rate: %.3f' %(m3+2, method3) + '%')


    print("method4")
    for m4 in range(len(AVH_ls)-1):
        V24= AVH_ls[m4]
        V34= AVH_ls[m4+1]
        method4 = (1-V24/V34)*100

        print('%d pressure rate: %.3f' %(m4+1, method4) + '%')



#     plt.figure(figsize=(30,30))
#     plt.imshow(img1)
