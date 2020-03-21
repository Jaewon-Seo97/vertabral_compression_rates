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

    file = filename[f]

    img = cv2.imread(file)

    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray1)
    img1 = img.copy()

    ret, thresh = cv2.threshold(gray1,127,255,0)

    contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # contours_image = cv2.drawContours(img1, contours, -1, (255,0,0),2)



    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray1)

    int_cent = centroids.astype(np.int32)

    pressure_loc = []

    AVH_ls = []

    for cnt in contours:

        epsilon = 0.09*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(img1, [approx], 0,(255,0,0), 1)


        # float_app = approx.astype(np.float32)
        # fd_app = np.empty((4,2))
        # id_app = np.empty((4,2))


        # for a in range(len(float_app)):
            # fd_app[a] = float_app[a][0]
            # id_app[a] = approx[a][0]

        [LT_x, LT_y] = [approx[1][0][0], approx[1][0][1]]
        [LB_x, LB_y] = [approx[2][0][0], approx[2][0][1]]
        [RT_x, RT_y] = [approx[0][0][0], approx[0][0][1]]
        [RB_x, RB_y] = [approx[3][0][0], approx[3][0][1]]

        AVH = math.sqrt((LT_x-LB_x)**2 + (LT_y-LB_y)**2)
        PVH = math.sqrt((RT_x-RB_x)**2 + (RT_y-RB_y)**2)

        AVH_ls.append(AVH)

        pressure_rate = (1-(AVH/PVH))*100

        pressure_loc.insert(0,pressure_rate)


        for i in range(len(approx)):
            cv2.putText(img1, str(i), (approx[i][0][0], approx[i][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 2)
        for i in range(nlabels):
            cv2.putText(img1, str(i), (int_cent[i][0], int_cent[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 2)


    count_bone = len(pressure_loc)

    max_index = pressure_loc.index(max(pressure_loc))




#method1
    print("method1")
    print('#'*15 + '-%d-' %(f+1) + '#'*15 )
    for n in range(len(pressure_loc)):

        pr = pressure_loc[n]
        print('%d pressure rate: %.3f' %(n+1, pr), end='')
        print('%')


    print('-'*30)
    print('max: bone%d, rate: %.3f' %(max_index+1,max(pressure_loc)), end='')
    print('%')

    print('#'*30)

    if max(pressure_loc)>=30:
        # print('#'*30)
        print('30%이상의 압박률')
        # print('#'*15 + '-%d-' %(f+1) + '#'*15 )
        plt.figure(figsize=(30,30))
        plt.imshow(img)
    else:
        pass


    #method2
    print("method2")
    for m2 in range(len(AVH_ls)-2):
        V12= AVH_ls[m2]
        V22= AVH_ls[m2+1]
        V32= AVH_ls[m2+2]
        method2 = ((V12+V32)/2-V22)/((V12+V32)/2)


        print('%d pressure rate: %.3f' %(m2+2, method2), end='')
        print("%")
#method3
    print("method3")
    for m3 in range(len(AVH_ls)-1):
        V13= AVH_ls[m3]
        V23= AVH_ls[m3+1]
        method3 = 1-V23/V13

        print('%d pressure rate: %.3f' %(m3+2, method3), end='')
        print("%")


#method4
    print("method4")
    for m4 in range(len(AVH_ls)-1):
        V24= AVH_ls[m4]
        V34= AVH_ls[m4+1]
        method4 = 1-V24/V34

        print('%d pressure rate: %.3f' %(m4+1, method4), end='')
        print("%")
