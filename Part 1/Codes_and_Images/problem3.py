#testimport

################ Section 1.0 Importing all the dependencies #############

import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import ipdb
from scipy import linalg as ln
from scipy import signal
import time

################### Section 2.a Loading and Displaying Images #############

# load image
image = cv2.imread('elephant.jpeg',1)
# display the image
cv2.imshow('image',image)
# wait indefinitely for a key to be pressed
cv2.waitKey(0)
#Destroy all windows after a key is pressed
cv2.destroyAllWindows()
 
################### Section 2.b Using matplot to display images ############

# plotting the image
img = plt.imshow(image)
plt.title('Prob 3.2.b plotting an openCV read Image by matplotlib is inverted')
plt.show()
# saving the image
c = cv2.imwrite('elephant_opencv.png',image)

############## Section 2.c Converting the BGR image to RGB image to display the original image###

#converting the image and mapping to RGB space
imgInverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#plotting the inverted image
plt.imshow(imgInverted)
plt.title('Prob 3.2.c converting the color space from BGR to RGB and plotting it using MATPLOTLIB')
plt.show()
c = cv2.imwrite('elephant_matplotlib.png', imgInverted)

#%%########## Section 2.d REading in the image as a Gray Scale image ##########
imgGray = cv2.imread('elephant.jpeg',0)
#plot the image by setting the colormap to Gray to plot a grayscale image
plt.imshow(imgGray, cmap='gray')
plt.title('prob 3.2.d read image as grayscale and plot in grayscale')
plt.show()

#%%################# Section 3.0 Cropping #####################################################
#cropping rows 380:950 and columns 100:550 to get baby elephant
imCrop = image[380:950, 100:550]
#plotting the cropped image in the correct RGB space
plt.imshow(cv2.cvtColor(imCrop, cv2.COLOR_BGR2RGB))
plt.title('prob 3.3: cropping the baby elephant')
plt.show()
c = cv2.imwrite('babyelephant.png',imCrop)

#%%################## Section 4.b Pixel Wise Arithmetic OPerations using numpy #############################

#adding 256 to every pixel of the RGB image loaded in section 2.c
imgHigh = imgInverted + 256
print('Prob 3.4.b: image data type ', imgHigh.dtype)
#convert the uint16 to uint8 image
imgReverted = np.uint8(imgHigh) 
plt.imshow(imgReverted)
plt.title('prob 3.4.b: adding 256 as a numpy arithmetic operation and reverting the result to uint8 datatype')
plt.show()

#%%################ Section 4.c pixelwise arithmetic using opencv##########################################
#splitting the image into bgr channels
b,g,r = cv2.split(image)
#adding 256 to each channel
b_add = cv2.add(b,256)
g_add = cv2.add(g,256)
r_add = cv2.add(r,256)
#merging the channels to form a new image
imgMerged = cv2.merge((b_add,g_add,r_add))
# plt.imshow(cv2.cvtColor(imgMerged, cv2.COLOR_BGR2RGB))
# plt.title('performing cv2.add to all the channels and merging gives')
# plt.show()

#%%################# Section  5.a and 5.b Resizing Images ########################################
imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#downsampling by a factor of 10

imgResize = cv2.resize(imgRGB, None, fx = 0.1, fy = 0.1, interpolation = cv2.INTER_AREA)
plot = plt.imshow(imgResize)
plt.title('shrinking the image 10 times')
plt.show()
c = cv2.imwrite('elephant_10xdown.png',cv2.cvtColor(imgResize,cv2.COLOR_RGB2BGR))

#%%####### Section 5.c upsampling the shrinked image ##################
#first upsample using nearest neighbor
imgUpsampledNearest = cv2.resize(imgResize, None, fx = 10, fy = 10, interpolation = cv2.INTER_NEAREST)
cv2.imwrite('elephant_10xup_nearest.png', cv2.cvtColor(imgUpsampledNearest, cv2.COLOR_RGB2BGR))
imgUpsampledCubic = cv2.resize(imgResize, None, fx = 10, fy = 10, interpolation = cv2.INTER_CUBIC)
cv2.imwrite('elephant_10xup_cubic.png', cv2.cvtColor(imgUpsampledCubic, cv2.COLOR_RGB2BGR))
plt.imshow(imgUpsampledNearest)
plt.title('prob 3.5.c: upsampled image by nearest neighbor interpoltion')
plt.show()
plt.imshow(imgUpsampledCubic)
plt.title('prob 3.5.c: upsampled image by bicubic interpoltion')
plt.show()

#%%######### Section 5.d calculating absdifference and total error #######

#absolute difference of upsampled image by nearest neighbor interpoltion from the original image
diffNearestArr = cv2.absdiff(imgUpsampledNearest,imgRGB)

#absolute difference of upsampled by bicubic interpolation image from the original image
diffCubicArr = cv2.absdiff(imgUpsampledCubic,imgRGB)

#summing all elements in the difference matrix to find out the errors in both the interpolation methods 
errorNearest = np.sum(diffNearestArr)
errorCubic = np.sum(diffCubicArr)
print('error by Nearest Neighbor interpolation is ', errorNearest)
print('error by Bicubic interpolation is ', errorCubic)

#plotting the difference
plt.imshow(diffNearestArr)
plt.title('prob 3.5.d: absolute difference of upsampled image by nearest neighbor interpoltion from the original image')
plt.show()

plt.imshow(diffCubicArr)
plt.title('prob 3.5.d: absolute difference of upsampled image by bicubic interpoltion from the original image')
plt.show()