# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 18:44:17 2022

@author: Amelia Waszkowska
"""
from cmath import nan
import numpy as np
import cv2 as cv
import glob
import sys
import os

# wczytanie ramek z pierwszej sekwencji
cv_img_1 = []   
for x in range(1, 540):
    path = "C:/Users/48668/Documents/Python/sekwencje/sequence_1/camera_1/frames/frame_%d.JPG" % x
    frame = cv.imread(path)
    height, width, layers = frame.shape
    size = (width, height)
    cv_img_1.append(frame)

img = cv.imread("C:/Users/48668/Documents/Python/sekwencje/sequence_1/camera_1/frames/frame_82")
cv.imshow('1st seq, 1st cam', cv_img_1[82])

# przekształcenie ramek w wideo (plik avi)
out_1 = cv.VideoWriter('seq1_cam1.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, size) 
for i in range(len(cv_img_1)):
    out_1.write(cv_img_1[i])
out_1.release()
   
cv_img_2 = []
for x in range(1, 540):
    path = "C:/Users/48668/Documents/Python/sekwencje/sequence_1/camera_2/frames/frame_%d.JPG" % x
    frame = cv.imread(path)
    height, width, layers = frame.shape
    size = (width, height)
    cv_img_2.append(frame)
    
out_2 = cv.VideoWriter('seq1_cam2.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, size) 
for i in range(len(cv_img_2)):
    out_2.write(cv_img_2[i])
out_2.release()
    
cv_img_3 = []
for x in range(1, 540):
    path = "C:/Users/48668/Documents/Python/sekwencje/sequence_1/camera_3/frames/frame_%d.JPG" % x
    frame = cv.imread(path)
    height, width, layers = frame.shape
    size = (width, height)
    cv_img_3.append(frame)
    
out_3 = cv.VideoWriter('seq1_cam3.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, size) 
for i in range(len(cv_img_3)):
    out_3.write(cv_img_3[i])
out_3.release()

#%% wczytanie ramek z drugiej sekwencji

cv_img_4 = []   
for x in range(1, 540):
    path = "C:/Users/48668/Documents/Python/sekwencje/sequence_2/camera_1/frames/frame_%d.JPG" % x
    frame = cv.imread(path)
    height, width, layers = frame.shape
    size = (width, height)
    cv_img_4.append(frame)

# przekształcenie ramek w wideo (plik avi)
out_4 = cv.VideoWriter('seq2_cam1.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, size) 
for i in range(len(cv_img_4)):
    out_4.write(cv_img_4[i])
out_4.release()
   
cv_img_5 = []
for x in range(1, 540):
    path = "C:/Users/48668/Documents/Python/sekwencje/sequence_2/camera_2/frames/frame_%d.JPG" % x
    frame = cv.imread(path)
    height, width, layers = frame.shape
    size = (width, height)
    cv_img_5.append(frame)
    
out_5 = cv.VideoWriter('seq2_cam2.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, size) 
for i in range(len(cv_img_5)):
    out_5.write(cv_img_5[i])
out_5.release()
    
cv_img_6 = []
for x in range(1, 540):
    path = "C:/Users/48668/Documents/Python/sekwencje/sequence_2/camera_3/frames/frame_%d.JPG" % x
    frame = cv.imread(path)
    height, width, layers = frame.shape
    size = (width, height)
    cv_img_6.append(frame)
    
out_6 = cv.VideoWriter('seq2_cam3.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, size) 
for i in range(len(cv_img_6)):
    out_6.write(cv_img_6[i])
out_6.release()

#%% detekcja piłki i wyznaczenie środka piłki (u,v)

def detekcja(cap, contour_size, circ_low, circ_high):
    detected = []
    idx_detected = []
    u = []
    v = []
    index = 0

    # zastosowanie algorytmu odejmowania tła w typie mixture of gaussians z włączoną detekcją cieni
    backSub = cv.createBackgroundSubtractorMOG2(detectShadows = True)
    kernel = None
    while(1):
        #przechwycenie ramki z wideo
        ret, frame = cap.read()
        if not ret:
            break
        #licznik indeksów ramek
        index += 1
        # wpisanie jako współrzędnych piłki None jako default dla każdej ramki
        u.append(None)
        v.append(None)
        
        # nałożenie maski odejmowania tła
        gmask = backSub.apply(frame)

        # ustawienie tresholdu aby pozbyć się cieni
        _, gmask = cv.threshold(gmask, 250, 255, cv.THRESH_BINARY)
        
        # aplikacja operacji morfologicznych w celu polepszenia maski
        gmask = cv.erode(gmask, kernel, iterations = 1)
        gmask = cv.dilate(gmask, kernel, iterations = 2)
        #znajduję konktury na ramce
        contours, _ = cv.findContours(gmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        frameCopy = frame.copy()
        
        for cnt in contours:
            perimeter = cv.arcLength(cnt, True)
            area = cv.contourArea(cnt)
            if perimeter == 0:
                break
            # definiuję okrągłość
            circularity = 4*np.pi*(area/(perimeter**2))
            # kontur ma być większy niż ustalona wartość i mieć odpowiednią cyrkularność
            if cv.contourArea(cnt) > contour_size and circularity < circ_high and circularity > circ_low :
                # obrysowuję wykryty obiekt
                x, y, width, height = cv.boundingRect(cnt)
                cv.rectangle(frameCopy, (x, y), (x + width, y + height),(0, 0, 255), 2)
                # tekst opisuje wykryty obiekt
                cv.putText(frameCopy, 'Ball', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv.LINE_AA)
                #dołączam ramkę i indeks do list wykrytych
                detected.append(frame)
                idx_detected.append(index)
                x_val = x + width/2
                y_val = y + height/2
                # w przypadku wykrycia piłki usuwam defaultowe None
                u.pop(index-1)
                v.pop (index-1)
                # i wpisuję konkretną wartość
                u.append(x_val)
                v.append(y_val)
                # break zabezpiecza przed wykryciem wielu obiektów
                break

        foregroundPart = cv.bitwise_and(frame, frame, mask=gmask)
        stacked = np.hstack((frame, foregroundPart, frameCopy))
        # pokazanie ramek wraz z ich przetworzeniami
        
        cv.imshow('all', cv.resize(stacked, None, fx=0.2, fy=0.35))
        # waitkey to escape, żeby wyjść z wyświetlania
        k = cv.waitKey(30) & 0xff
        if k == 27:
            cv.destroyAllWindows()
            break
    return u, v, idx_detected

# pierwsza sekwencja
cap_1 = cv.VideoCapture('seq1_cam1.avi')
u_1, v_1, idx_detected_1 = detekcja(cap_1, 600, 0.45, 1.25)

cap_2 = cv.VideoCapture('seq1_cam2.avi')
u_2, v_2, idx_detected_2 = detekcja(cap_2, 650, 0.50, 1.25)

cap_3 = cv.VideoCapture('seq1_cam3.avi')
u_3, v_3, idx_detected_3 = detekcja(cap_3, 600, 0.55, 1.25)

# druga sekwencja
cap_4 = cv.VideoCapture('seq2_cam1.avi')
u_4, v_4, idx_detected_4 = detekcja(cap_4, 600, 0.45, 1.25)

cap_5 = cv.VideoCapture('seq2_cam2.avi')
u_5, v_5, idx_detected_5 = detekcja(cap_5, 650, 0.50, 1.25)

cap_6 = cv.VideoCapture('seq2_cam3.avi')
u_6, v_6, idx_detected_6 = detekcja(cap_6, 600, 0.55, 1.25)


#%% inner calibration 1st sequence

objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:0.84:8j,0:0.60:6j].T.reshape(-1,2)
objpoints_1 = []
imgpoints_1 = []
objpoints_2 = []
imgpoints_2 = []
objpoints_3 = []
imgpoints_3 = []
cv_calib_1 = [] 
cv_calib_2 = [] 
cv_calib_3 = [] 
# wymiary tablicy kalibracyjnej, 6x8 kół
patternSize = [8,6]   
for x in range(1, 35):
    path = "C:/Users/48668/Documents/Python/sekwencje/sequence_1/camera_1/calib/frame_%d.JPG" % x
    frame = cv.imread(path)
    height, width, layers = frame.shape
    size = (width, height)
    cv_calib_1.append(frame)    
    # znajdowanie wzorca    
    patternfound, centers = cv.findCirclesGrid(frame, patternSize)
    # jeśli znaleziony, dołączam do listy objectpoints i środki
    if patternfound == True:
        objpoints_1.append(objp)
        imgpoints_1.append(centers)

# kalibracja wewnętrzna, wyznaczenie macierzy kamery, współczynnika dystorsji, dist, wektora rotacji i translacji   
ret_1, mtx_1, dist_1, rvecs_1, tvecs_1 = cv.calibrateCamera(objpoints_1, imgpoints_1, size, None, None)

for x in range(1, 35):
    path = "C:/Users/48668/Documents/Python/sekwencje/sequence_1/camera_2/calib/frame_%d.JPG" % x
    frame = cv.imread(path)
    height, width, layers = frame.shape
    size = (width, height)
    cv_calib_2.append(frame)        
    patternfound, centers = cv.findCirclesGrid(frame, patternSize)
    if patternfound == True:
        objpoints_2.append(objp)
        imgpoints_2.append(centers)
    
ret_2, mtx_2, dist_2, rvecs_2, tvecs_2 = cv.calibrateCamera(objpoints_2, imgpoints_2, size, None, None)

for x in range(1, 35):
    path = "C:/Users/48668/Documents/Python/sekwencje/sequence_1/camera_3/calib/frame_%d.JPG" % x
    frame = cv.imread(path)
    height, width, layers = frame.shape
    size = (width, height)
    cv_calib_3.append(frame)        
    patternfound, centers = cv.findCirclesGrid(frame, patternSize)
    if patternfound == True:
        objpoints_3.append(objp)
        imgpoints_3.append(centers)
    
ret_3, mtx_3, dist_3, rvecs_3, tvecs_3 = cv.calibrateCamera(objpoints_3, imgpoints_3, size, None, None)

#%% outer calibration
def outer_calib(i_Points, mtx, dist):
    """_summary_

    Args:
        i_Points : pixel coordinates of points on an image, corresponding to object points
        mtx : camera matrix (from inner calibration)
        dist : distortion coefficient (from inner calibration)

    Returns:
        retval
        rvec_mat : rotation matrix of a camera
        tvec : translation vector of a camera
    """
    # object points - punkty z boiska, przedstawione w prezentacji
    o_Points = np.array([12,0,0,9,9,0,12,9,0,18,9,0,18,0,0], dtype=np.float32)
    objectPoints = o_Points.reshape(5,3,1)
    flag = cv.SOLVEPNP_ITERATIVE

    imagePoints = i_Points.reshape(5,2,1)
    retval, rvec, tvec = cv.solvePnP(objectPoints, imagePoints, mtx, dist, flags=flag)
    # przekształcenie wektora w macierz za pomocą funkcji Rodrigues
    rvec_mat, _ = cv.Rodrigues(rvec)

    return retval, rvec_mat, tvec

i_Points_1 = np.array([92,832,572,476,932,489,1846,528,1854,1017], dtype=np.float32)
retval_1, rvec_mat_1, tvec_1 = outer_calib(i_Points_1, mtx_1, dist_1)

i_Points_2 = np.array([92,832,572,476,932,489,1846,528,1854,1017], dtype=np.float32)
retval_2, rvec_mat_2, tvec_2 = outer_calib(i_Points_2, mtx_2, dist_2)

i_Points_3 = np.array([92,832,572,476,932,489,1846,528,1854,1017], dtype=np.float32)
retval_3, rvec_mat_3, tvec_3 = outer_calib(i_Points_3, mtx_3, dist_3)

# %% wyznaczenie punktów znormalizowanych parametrami z kalibracji oraz alf (directional vector)
# i macierzy R inv, T inv, czyli w kodzie odpowiednio rmat_inv_NumCam i tvec_inv_NumCam

def alfa_i_norma(mtx, rvec_mat, tvec, u, v):
    c_x_1 = mtx[0][2]
    c_y_1 = mtx[1][2]
    f_x_1 = mtx[0][0]
    f_y_1 = mtx[1][1]

    rmat_inv = rvec_mat.T
    tvec_inv = np.matmul(-rmat_inv,tvec)

    P_norm = np.zeros((len(u),3))
    alfa = np.zeros((len(u),3))
    for i in range (len(u)):
        # warunek bo gdy piłka nie jest wykryta, to dla tej ramki wpisuję None i tak ma pozostać (else)
        if u[i] is not None and v[i] is not None:
            P_norm[i,0] = (u[i] - c_x_1)/f_x_1
            P_norm[i,1] = (v[i] - c_y_1)/f_y_1
            P_norm[i,2] = 1
        
        else:
            P_norm[i,:] = None
        alfa = np.matmul(P_norm, rmat_inv)

    return P_norm, alfa, tvec_inv

# pierwsza sekwencja
P_norm_1, alfa_1, tvec_inv_1 = alfa_i_norma(mtx_1, rvec_mat_1, tvec_1, u_1, v_1)
P_norm_2, alfa_2, tvec_inv_2 = alfa_i_norma(mtx_2, rvec_mat_2, tvec_2, u_2, v_2)
P_norm_3, alfa_3, tvec_inv_3 = alfa_i_norma(mtx_3, rvec_mat_3, tvec_3, u_3, v_3)

# druga sekwencja
P_norm_4, alfa_4, tvec_inv_4 = alfa_i_norma(mtx_1, rvec_mat_1, tvec_1, u_4, v_4)
P_norm_5, alfa_5, tvec_inv_5 = alfa_i_norma(mtx_2, rvec_mat_2, tvec_2, u_5, v_5)
P_norm_6, alfa_6, tvec_inv_6 = alfa_i_norma(mtx_3, rvec_mat_3, tvec_3, u_6, v_6)

#%% wyznaczenie punktów
def point3D(A, B, alfa_prim, alfa_bis):
    """
    Args:
        A : tvec_inv_NumCamFirst, which is a positional vector,
            a matrix multiplication of negative inverted rotation
            matrix of a camera and its translation vector
        B : tvec_inv_NumCamSecond, same

        alfa_prim: alfa_NumCamFirst, which is a directional vector,
                   a matrix multiplication of normalised points from
                   frames and inverted rotation matrix of a camera
        alfa_bis: alfa_NumCamSecond, same

    Returns:
        Point3D : a matrix of 3D point coordinates for every frame
    """

    c = B - A

    ab = np.zeros(540)
    ac = np.zeros((540))
    bc = np.zeros((540))
    aa = np.zeros((540))
    bb = np.zeros((540))
    ab = np.zeros((540))
    D = np.zeros((540,3))
    E = np.zeros((540,3))
    Point3D = np.zeros((540,3))

    A_resh = np.reshape(A,(1,3))
    B_resh = np.reshape(B,(1,3))

    for i in range (539):
        ab[i] = np.dot(alfa_prim[i,:], alfa_bis[i,:])
        ac[i] = np.dot(alfa_prim[i,:], c[:,:])
        bc[i] = np.dot(alfa_bis[i,:], c[:,:])
        aa[i] = np.dot(alfa_prim[i,:], alfa_prim[i,:])
        bb[i] = np.dot(alfa_bis[i,:], alfa_bis[i,:])

        D[i,:] = A_resh + (alfa_prim[i,:])*(ac[i]*bb[i]-ab[i]*bc[i])/(aa[i]*bb[i]-ab[i]*ab[i])
        E[i,:] = B_resh + (alfa_bis[i,:])*(ab[i]*ac[i]-bc[i]*aa[i])/(aa[i]*bb[i]-ab[i]*ab[i])

        Point3D[i,:] = 0.5*(D[i,:]+E[i,:])

    return Point3D

# dla pierwszej sekwencji
Point_Cam12 = point3D(tvec_inv_1, tvec_2, alfa_1, alfa_2)
Point_Cam13 = point3D(tvec_inv_1, tvec_3, alfa_1, alfa_3)
Point_Cam23 = point3D(tvec_inv_2, tvec_3, alfa_2, alfa_3)

Point = (Point_Cam12 + Point_Cam13 + Point_Cam23)/3
np.savetxt('C:/Users/48668/Documents/Python/coordinates_seq1.csv', Point, delimiter=',', header='X, Y, Z')

# dla drugiej sekwencji
Point_Cam12 = point3D(tvec_inv_1, tvec_2, alfa_4, alfa_5)
Point_Cam13 = point3D(tvec_inv_1, tvec_3, alfa_4, alfa_6)
Point_Cam23 = point3D(tvec_inv_2, tvec_3, alfa_5, alfa_6)

Point = (Point_Cam12 + Point_Cam13 + Point_Cam23)/3
np.savetxt('C:/Users/48668/Documents/Python/coordinates_seq2.csv', Point, delimiter=',', header='X, Y, Z')








