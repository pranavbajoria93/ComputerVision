
# -*- coding: utf-8 -*-

"""
Created on Sun Sep 15 21:06:08 2019

Source citation: https://github.com/alisaaalehi/convolution_as_multiplication/blob/master/ConvAsMulExplained.pdf

@author: Pranav
"""

#%%######################## IMporting dependencies ######################
import numpy as np

from matplotlib import pyplot as plt
from scipy import linalg as ln
from scipy import signal
import time
import cv2

def vector2matrix(vec, opShape):
    opRow, opCol = opShape
    op = np.zeros(opShape, dtype = vec.dtype)
    for i in range(opRow):
        start = i*opCol
        end = start + opCol
        op[i, :] = vec[start:end]
        
    return op


def matrix2vector(mat):
    matRow, matCol = mat.shape
    outputVector = np.zeros(matRow*matCol,dtype=mat.dtype)
    for i,row in enumerate(mat):
        startPoint = i*matCol
        endPoint = startPoint + matCol
        outputVector[startPoint:endPoint] = row
        
    return outputVector

####################### Function for convolution of 2 matrices ################
def conv2dmatrix(Img, Filter):
    
    #store the time the code starts
    start = time.time()
    
    #Inverting the filter around the vertical axis for convolution
    Filter = np.flip(Filter)
    
    #Calculating the size of the output image based on the filter and input image size given by
    #OutputRow,OutputColumn = filterRow+imgRow-1 , filterColumn+imgCol-1
    imgRow, imgCol = Img.shape
    filterRow, filterCol = Filter.shape
    
    #output image dimensions
    resultImgRow = imgRow+filterRow-1
    resultImgCol = imgCol+filterCol-1
    
    #padding zeros to the filter to replicate the output matrix dimensions
    padFilter = np.pad(Filter, ((resultImgRow-filterRow,0),(0,resultImgCol-filterCol)),'constant',constant_values=0)
    #print(padFilter)
    
    ###########################Create individual toeplitz matrices##########
    #to create toeplitz matrices, we need to make sure that each toeplitz matrix is of the same no. of columns as the input image
    
    #creating a list for stacking the toeplitz matrices to build the final H matrix
    toeplitzStack = []
    #since we start to fill up matrices from the last row of the padded filter, we iterate from last row to first
    for i in range(padFilter.shape[0]-1,-1,-1):
        # extract column that has to be passed to the toepliz function 
        col = padFilter[i,:]
        #initiate first row of toeplitz with zeros other than the first value
        row = np.r_[col[0], np.zeros(imgCol-1)]
        #call toeplitz function with the set parameters
        tMatrix = ln.toeplitz(col,row)
        #stack the matrix in the list
        toeplitzStack.append(tMatrix)

    
    #the toeplitz matrix of the small toeplitz matrices can have no. of columns = no. of rows in the Input image matrix 
    #and the number of small toeplitz matrices in each column should be same as the output image's no. of rows
    colBigIndices = range(1,resultImgRow+1)
    rowBigIndices = np.r_[colBigIndices[0],np.zeros(imgRow-1,dtype=int)]
    
    #generating the indices in the Bigger Toeplitz matrix to fill with their respective smaller toeplitz from the stack.
    
    indicesBig = ln.toeplitz(colBigIndices,rowBigIndices)
    
    #Finally generate the big toeplitx matrix
    
    #First calculating the shape and size
    
    rowsBig = tMatrix.shape[0]*indicesBig.shape[0]
    colBig = tMatrix.shape[1]*indicesBig.shape[1]
    toeplitzBigShape = [rowsBig,colBig]
    toeplitzBigMatrix = np.zeros(toeplitzBigShape)
    

    #each small toeplitz matrice's height and width
    small_h, small_w = tMatrix.shape[0], tMatrix.shape[1]
    for i in range(indicesBig.shape[0]):
        for j in range(indicesBig.shape[1]):
            iStart = i * small_h
            jStart = j * small_w
            iEnd = iStart + small_h
            jEnd = jStart + small_w
            toeplitzBigMatrix[iStart:iEnd, jStart:jEnd] = toeplitzStack[indicesBig[i,j]-1]
    
    #Now convert the input image to a column vector to enable matrix multiplication
    inputImgVectorized = matrix2vector(Img)
    
    #now multiply the vectorized input with the big toeplitz to get result
    resultImgVec = np.matmul(toeplitzBigMatrix, inputImgVectorized)
    
    resultImgMat = vector2matrix(resultImgVec, padFilter.shape)
    
    timeTaken = time.time() - start
    resultByFunc = signal.convolve(Img, Filter, mode ='full')
    
# Calculating error
    difference = resultImgMat - resultByFunc
    error = np.sum(difference)
    return resultImgMat, timeTaken, error
### Problem 4.b Computing i*h by matrix multiplication

   
#%%###################### Main ##############################    
#image input as the given matrix
mat = np.array([[1,2,3],[4,5,6],[7,8,9]])

#filter or kernel to convolute
kernel = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
result, timeElapsed, errorMat = conv2dmatrix(mat,kernel)
print('the result by matrix mul of the given array is \n', result)
print('\n the time taken is ', timeElapsed)
print('\n the error in convolving the matrix is ', errorMat)

### Problem 4.c Generalising and testing conv2dmatrix for an image

#test it on the image
elephant = cv2.imread('elephant_10xdown.png',0)
resultImg, timeImg, errorImg = conv2dmatrix(elephant, kernel)
plt.imshow(resultImg, cmap='gray')
plt.show()
print('the result by matrix mul of the elephant image is \n', resultImg)
print('\n the time taken for image convolution is', timeImg)
print('\n the error in convolving the Image is ', errorImg)

