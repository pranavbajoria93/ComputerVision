# Importing all the dependencies

import numpy as np
from matplotlib import pyplot as plt
import cv2
import ipdb
import scipy

# function for performing corner detection
def detectHarris(Img):
    row, col = Img.shape
    iX = cv2.Sobel(Img,cv2.CV_64F,1,0,ksize = 3)
    iXNorm = iX - np.mean(iX)
    iY = cv2.Sobel(Img,cv2.CV_64F,0,1,ksize = 3)
    iYNorm = iY - np.mean(iY)
    iX_2 = iXNorm*iXNorm
    iY_2 = iYNorm*iYNorm
    iXY = iXNorm*iYNorm

    iX_2Norm = cv2.GaussianBlur(iX_2, (3,3), 2)
    iY_2Norm = cv2.GaussianBlur(iY_2, (3,3), 2)
    iXYNorm = cv2.GaussianBlur(iXY, (3,3), 2)
    R = np.zeros((row, col))
    maxCorner = 0
    cornerIndices = list()
    for i in range(row):
        for j in range(col):
            M = np.array([[iX_2Norm[i,j], iXYNorm[i,j]],[iXYNorm[i,j], iY_2Norm[i,j]]], dtype = np.float64)
            R[i, j] = np.linalg.det(M) - 0.04 * np.power(np.trace(M),2)
            if (R[i,j]>maxCorner):
                maxCorner = R[i, j]
    for k in range(row-1):
        for l in range(col-1):
            if(R[k,l]> 0.25*maxCorner and R[k,l]>R[k-1,l-1] and R[k,l]> R[k-1, l+1] and R[k,l]>R[k+1, l-1]) and R[k,l]>R[k+1, l+1] and R[k,l]>R[k, l-1] and R[k,l]>R[k-1, l] and R[k,l]>R[k, l+1] and R[k,l]>=R[k+1, l] :
    #            print("index found \n", k, l) and R[k,l]>R[k, l-1] and R[k,l]>R[k-1, l] and R[k,l]>R[k, l+1] and R[k,l]>R[k+1, l]
                cornerIndices.append([l,k])
                
    
    return cornerIndices


######## Section 4.1 ##########
img = np.zeros((300,300), np.uint8)
polyPoints = np.array([[180, 60], [120,75], [100, 135], [200, 165]], dtype = np.int32)
originalImg = cv2.fillPoly(img, [polyPoints], 1)
# img[125:175, 125:175] = np.ones((50,50), np.uint8)
plt.imshow(originalImg, 'gray')
#center = img.shape

rotationMat = cv2.getRotationMatrix2D((150,150), 45, 1)
rotatedImg = cv2.warpAffine(originalImg, rotationMat, originalImg.shape)
translationMatrix = np.float32([[1,0,30],[0,1,100]])
translatedImg = cv2.warpAffine(rotatedImg, translationMatrix, img.shape)
plt.imshow(translatedImg, 'gray')

#cornersOriginal = detectHarris(originalImg)
cornersTransformed = detectHarris(translatedImg)

print(cornersTransformed)