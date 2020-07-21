# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 21:58:04 2019

@author: User
"""
#%%############## Section 1.0 Importing all the dependencies ##############################

import cv2
import numpy as np
from matplotlib import pyplot as plt
import ipdb

#%%############## Section 2.0 Fourier transform of the image #############

#reading in the image as a grayscale
img = cv2.imread('elephant.jpeg',0)

plt.imshow(img, cmap='gray')
plt.show()
#
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
fourierImg = np.copy(fshift)
magnitudeSpectrum = 20*np.log(np.abs(fshift))
phaseSpectrum = np.angle(fshift)
plt.imshow(magnitudeSpectrum, cmap='gray')
plt.title('Magnitude'), plt.xticks([]), plt.yticks([])
plt.show()
plt.imshow(phaseSpectrum, cmap='gray')
plt.title('Phase'), plt.xticks([]), plt.yticks([])
plt.show()

#%%############## Section 2.1 Applying High Pass filter ###################################

#making the low frequency elements in the frequesncy domain imgae 0 to remove them
#getting the midpoint of the image to iterate 40 up and down 
fs = np.copy(fshift)
fshift[(img.shape[0]//2)-40 : (img.shape[0]//2)+40, (img.shape[1]//2)-40 : (img.shape[1]//2)+40] = 0

#performing inverse fourier transform
fIshift = np.fft.ifftshift(fshift)
imgHighPass = np.fft.ifft2(fIshift)
imgHighPass = np.absolute(imgHighPass)

#plotting the highPass Image
plt.imshow(imgHighPass, cmap='gray')
plt.title('High Pass filtered Image')
plt.show() 

#%%############## Section 2.1 Applying High Pass filter ###################################

#making the low frequency elements in the frequesncy domain imgae 0 to remove them
#getting the midpoint of the image to iterate 40 up and down 

imgMask = np.zeros((1000,1500), np.uint8)

imgMask[(img.shape[0]//2)-50 : (img.shape[0]//2)+50, (img.shape[1]//2)-50 : (img.shape[1]//2)+50] = 1

fs = fs*imgMask
fIshiftLowPass = np.fft.ifftshift(fs)
imgLowPass = np.fft.ifft2(fIshiftLowPass)
imgLowPass = np.absolute(imgLowPass)

#plotting the highPass Image
plt.imshow(imgLowPass, cmap='gray')
plt.title('Low Pass filtered Image')
plt.show() 

#%%############### Section 2.2 Combining phase and magnitude of 2 different images ############

imgLenna = cv2.imread('Lenna.jpg',0)
#plt.imshow(imgLenna, cmap = 'gray')
#plt.show()
#print(np.shape(imgLenna))
imgZebra = cv2.imread('zebra.jpg',0)
#print(np.shape(imgZebra))

fLenna = np.fft.fftshift(np.fft.fft2(imgLenna))
#extracting magnitude and phase from the fourier transformed images
magnitudeLenna = np.abs(fLenna)
phaseLenna = np.angle(fLenna)

plt.imshow(20*np.log(magnitudeLenna), cmap='gray')
plt.title('Lenna\'s magnitude')
plt.show()

plt.imshow(phaseLenna, cmap='gray')
plt.title('Lenna Phase')
plt.show()


fZebra = np.fft.fftshift(np.fft.fft2(imgZebra))
#extracting magnitude and phase from the fourier transformed images
magnitudeZebra = np.abs(fZebra)
phaseZebra = np.angle(fZebra) 

plt.imshow(20*np.log(magnitudeZebra), cmap='gray')
plt.title('Zebra\'s magnitude')
plt.show()

plt.imshow(phaseZebra, cmap='gray')
plt.title('Zebra Phase')
plt.show()

#combining the phase of 1 and magnitude of other by mag*exp(j*phase)
LennaMagZebraPhase = np.multiply(magnitudeLenna, np.exp(1j*phaseZebra))

#extracting the real part of the array to display the image
LennaMagZebraPhaseImg = np.real(np.fft.ifft2(np.fft.ifftshift(LennaMagZebraPhase)))
plt.imshow(LennaMagZebraPhaseImg, cmap='gray')
plt.title('Phase swapping: Lenna Magnitude Zebra Phase')
plt.show()

ZebraMagLennaPhase = np.multiply(magnitudeZebra, np.exp(1j*phaseLenna))

#extracting the real part of the array to display the image
ZebraMagLennaPhaseImg = np.real(np.fft.ifft2(np.fft.ifftshift(ZebraMagLennaPhase)))
plt.imshow(ZebraMagLennaPhaseImg, cmap='gray')
plt.title('Phase swapping: Zebra Magnitude Lenna Phase')
plt.show()

#%%################## Section 2.3 Hybrid images ##############################

imgCat = cv2.imread('cat.jpg',1)
imgDog = cv2.imread('dog.jpg',1)
dogBlur = cv2.GaussianBlur(imgDog,(45,45),11,cv2.BORDER_DEFAULT)
catBlur = cv2.GaussianBlur(imgCat,(67,67),5,cv2.BORDER_DEFAULT)
catSharp = cv2.subtract(imgCat, catBlur)
plt.imshow(dogBlur)
plt.show()
plt.imshow(catSharp)
plt.show()
hybrid = dogBlur + catSharp
plt.imshow(hybrid)
plt.show()