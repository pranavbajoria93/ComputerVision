#!/usr/bin/env python
# coding: utf-8

# # Problem 6 Multiresolution Blending using Laplacian/ Gaussian Pyramids

# In[109]:


################ Section 1.0 Importing all the dependencies ##############################

import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd 


# In[171]:


#function for multiresolution Blending
def multiblend(img1, img2):
    mask = np.zeros(img1.shape,np.float64)
    mask[:,0:(mask.shape[1]//2)] = 1
    levels = 4
    kern = (31,3)
    stdDev = 30
    maskPyramid = GaussianPyramid(mask, levels, kern, stdDev)
    lapOrange = LaplacianPyramid(img1,levels,kern, stdDev)
    lapApple = LaplacianPyramid(img2,levels,kern, stdDev)
    blendedLapPyramid = []
    for i in range(levels):
        blendedLapPyramid.append(cv2.add(maskPyramid[i]*lapOrange[i],(1-maskPyramid[i])*lapApple[i]))
    blendedReconPyramid = laplaceReconstruct(blendedLapPyramid, kern, stdDev)
    return blendedReconPyramid[len(blendedReconPyramid)-1]


# In[143]:


#Gaussian Pyramid
def GaussianPyramid(image, levels, kernel_tup, stdDev):
    gaussianPyramid = []
    
    for i in range(levels):
        gaussianPyramid.append(image)
        blurredImg = cv2.GaussianBlur(image,kernel_tup, stdDev, cv2.INTER_AREA)
        image = cv2.resize(blurredImg,(image.shape[0]//2,image.shape[1]//2),fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
        
    return gaussianPyramid


# In[157]:


# function for Laplacian Pyramid
def LaplacianPyramid(image, levels, kernTup, sDev):
    laplacianStack = []
    #while downsampling
    for i in range(levels):
        
        #G = blurring and downsampling
        blurredImg = cv2.GaussianBlur(image,kernTup,sDev,cv2.BORDER_DEFAULT)
        shrinkedImg = cv2.resize(blurredImg,(image.shape[0]//2,image.shape[1]//2),fx=0.5,fy=0.5)
        
        #F = blurring and upsampling
        blurShrinked = cv2.GaussianBlur(shrinkedImg,kernTup,sDev,cv2.BORDER_DEFAULT)
        upSampled = cv2.resize(blurShrinked,(image.shape[0],image.shape[1]),fx=2,fy=2)
        
        #Laplacian image = (I-FiGi)Img
        laplacianImage = cv2.subtract(image, upSampled)        
        if(i==levels-1):
            laplacianStack.append(blurredImg)
        else:
            laplacianStack.append(laplacianImage)

        image = shrinkedImg
    return laplacianStack


# In[158]:


def laplaceReconstruct(lapStack, kernTup, sDev):
    reconstrcutedPyramid = []
    image = lapStack[len(lapStack)-1]    
    for i in range(len(lapStack)-1,0,-1):
        reconstrcutedPyramid.append(image)
        imgAtUpperLevel = lapStack[i-1]
        
        #F = blurring and upsampling
        blurShrinked = cv2.GaussianBlur(image,kernTup,sDev,cv2.BORDER_DEFAULT)
        upSampled = cv2.resize(blurShrinked,(imgAtUpperLevel.shape[0],imgAtUpperLevel.shape[1]),fx=2,fy=2)
        
        #Laplacian image = (I-FiGi)Img
        image = cv2.add(imgAtUpperLevel, upSampled)
    reconstrcutedPyramid.append(image)
    return reconstrcutedPyramid
        


# In[159]:


#function for multiresolution Blending
def multiblend(img1, img2):
    mask = np.zeros(img1.shape,np.float64)
    mask[:,0:(mask.shape[1]//2)] = 1
    levels = 4
    kern = (31,3)
    stdDev = 30
    maskPyramid = GaussianPyramid(mask, levels, kern, stdDev)
    lapOrange = LaplacianPyramid(img1,levels,kern, stdDev)
    lapApple = LaplacianPyramid(img2,levels,kern, stdDev)
    blendedLapPyramid = []
    for i in range(levels):
        blendedLapPyramid.append(cv2.add(maskPyramid[i]*lapOrange[i],(1-maskPyramid[i])*lapApple[i]))
    blendedReconPyramid = laplaceReconstruct(blendedLapPyramid)
    return blendedReconPyramid[len(blendedReconPyramid)-1]


# In[160]:


# load image
img = cv2.imread('apple.jpeg',1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title('original image')
plt.show()


# ## Part a Gaussian and Laplacian Pyramids

# In[161]:


appleGaussPyramid = GaussianPyramid(img, 4, (35,35), 11)
for i in range(len(appleGaussPyramid)):
    plt.imshow(cv2.cvtColor(appleGaussPyramid[i], cv2.COLOR_BGR2RGB))
    plt.title('Image at Gaussian pyramid level '+str(i))
    plt.show()


# In[162]:


lap = LaplacianPyramid(img,4, (45,45), 7)
for i in range(len(lap)):
    plt.imshow(cv2.cvtColor(lap[i], cv2.COLOR_BGR2RGB))
    plt.title('Image at Laplacian pyramid level '+str(i))
    plt.show()
recon = laplaceReconstruct(lap, (45,45), 7)
for j in range(len(recon)):
    plt.imshow(cv2.cvtColor(recon[j], cv2.COLOR_BGR2RGB))
    plt.title('Reconstructed Image at pyramid level '+str(j))
    plt.show()


# In[ ]:





# # part b

# In[163]:


apple = cv2.imread('apple.jpeg',1)
orange = cv2.imread('orange.jpeg',1)
plt.imshow(cv2.cvtColor(apple,cv2.COLOR_BGR2RGB))
plt.title('apple original image')
plt.show()
plt.imshow(cv2.cvtColor(orange,cv2.COLOR_BGR2RGB))
plt.title('orange original image')
plt.show()


# In[164]:


#converting them to double precision
appleDouble = np.float64(apple)
orangeDouble = np.float64(orange)


# # part c Creating a mask

# In[165]:


#the mask has same size as apple image
mask = np.zeros(appleDouble.shape,np.float64)
mask[:,0:(mask.shape[1]//2)] = 1
maskAbs = cv2.convertScaleAbs(mask)
maskAbs = cv2.normalize(maskAbs, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
plt.imshow(maskAbs)
plt.show()


# # part d: Direct Blending

# In[166]:


#I = (1-M)*I1 + M*I2

# orangeMasked = cv2.bitwise_and(orangeDouble,orangeDouble,mask=mask)
orangeMasked = mask * orangeDouble 
appleMasked = (1-mask)*appleDouble
orapple = orangeMasked+appleMasked
plt.imshow(cv2.cvtColor(np.uint8(orapple),cv2.COLOR_BGR2RGB))
plt.show()


# # part e Alpha Blending

# In[167]:


# applying gaussian filter to the mask
filteredMask = cv2.GaussianBlur(mask, (33,33), 45)
alphaOrapple = filteredMask*orangeDouble + (1-filteredMask)*appleDouble
plt.imshow(cv2.cvtColor(np.uint8(alphaOrapple),cv2.COLOR_BGR2RGB))
plt.title('alpha Blending')
plt.show()


# # part f: Multiresolution Blending

# In[172]:


# plt.imshow(cv2.cvtColor(np.uint8(mask),cv2.COLOR_BGR2RGB))
orAppleMulti = multiblend(orangeDouble,appleDouble)
orAppleAbs = cv2.convertScaleAbs(orAppleMulti)
orAppleAbs = cv2.normalize(orAppleAbs, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
plt.imshow(cv2.cvtColor(orAppleAbs,cv2.COLOR_BGR2RGB))
plt.title('Multiresolution Blending')
plt.show()


# In[ ]:




