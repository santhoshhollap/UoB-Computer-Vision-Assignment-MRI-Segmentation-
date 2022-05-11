'''
File			: Final3D.py
Application		: 3D MRI segmentation 
Author			: Santhosh Holla Prakash
Creation Date	: 13/05/2022
description		: This .py file will read the input 3D MRI image (which is normalised to 0-255) and does 
				  process it to provide the segmented image. 
Usage			: python3 Finay3D.py <input_path> <output_path>
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
from Finalcluster import *
from threshold import *
from skimage import measure
from skimage.segmentation import (checkerboard_level_set,chan_vese)
from skimage.filters import threshold_multiotsu

class Main:
	'''
	This class is responsible for processing the 3D MRI image,based on the 
	method that is configured in constructor and output the colored segmented image
	'''
	def __init__(self): 
		'''
		Constructor to initialise the input path, configure the method to be experimented
		'''
		self.files = sorted(os.listdir(sys.argv[1]))
		self.class12Extraction = "multiotsu_measureLable_dilate"
		self.matterMethod = "fuzzymultiotsu" #fuzzymultiotsu or multiotsu 
		self.custom()

	def fillColor(self,clustImg):
		'''
		Function to provide the colored image as per the color which is seen in question document
		Arguments -
			clustImg      : A final segmented gray MRI image 
		Returns -
			clustImg : A final segmented 3 channeled color (as seen in question) image. 
		'''
		clustImg[np.where((clustImg==[0,0,0]).all(axis=2))] = [139,0,0] # class 0
		clustImg[np.where((clustImg==[50,50,50]).all(axis=2))] = [230,100,44] # class 1
		clustImg[np.where((clustImg==[80,80,80]).all(axis=2))] = [0,255,255]# class 3
		clustImg[np.where((clustImg==[150,150,150]).all(axis=2))] = [50,50,255]# class 4
		clustImg[np.where((clustImg==[190,190,190]).all(axis=2))] = [250,250,28]# class 2
		clustImg[np.where((clustImg==[250,250,250]).all(axis=2))] = [0,0,128]# class 5
		return clustImg

	def getMask3D(self,img):
		'''
		Function to does the operation similar to opencv drawcontour, it with specified color
		Arguments -
			img      : A image with a closed boundary
		Returns -
			im_out : A image with a filled closed boundary with specified color
		'''
		im_out = []
		for i in range(0,10):
			im_floodfill = img[i].copy()
			h, w = img[i].shape[:2]
			mask = np.zeros((h+2, w+2), np.uint8)
			cv2.floodFill(im_floodfill, mask, (0,0), 255)
			im_floodfill_inv = cv2.bitwise_not(im_floodfill)
			im_out.append(img[i] | im_floodfill_inv)
		return np.asarray(im_out)

	def segmentInnerPart(self,segmentedImg3,img):
		'''
		This funtion performs the segmentation of inner matter mask as per the configured method in init
		Arguments -
			segmentedImg3      : Mask of the inner matter - inner brain region 
			img 			   : original grayscale image stack
		Returns -
			clustImg : Inner brain region segmented image 
		'''
		#optimal algorithm, fuzzy and multiotsu combined algorithm
		if self.matterMethod == "fuzzymultiotsu":
			clust1 = fuzzyCluster(segmentedImg3*img,4)
			clustImg1 = clust1.cluster3D()
			clust2 = Threshold(np.uint8(segmentedImg3*img)*255)
			clustImg2 = clust2.multiOtsu()
			clustImg = clust2.handle(clustImg2,clustImg1)
			return clustImg
		# standalone multiotsu algorithm
		elif self.matterMethod == "multiotsu":
			clustImg = Threshold(np.uint8(segmentedImg3*img)*255)
			clustImg = clustImg.multiOtsu()
			return clustImg

	def otsuMeasureLableDilate(self,img):
		'''
		This funtion separate the inner matter from the outer ring using skimage's multiotsu method and skimage's
		measure label connective detection method and finally segments class 0,1,and 2
		Arguments - 
			img 			   : original grayscale image stack
		Returns -
			segmentedImg1 	   : class 1 mask
			segmentedImg2 	   : class 2 mask
			segmentedImg3 	   : inner matter mask - class345
			segmentedImg3ab    : For boosting accuracy of class 3, extra periphery region mask
		'''
		thresholds_brain = threshold_multiotsu(img,classes=2)
		regions_brain = np.digitize(img, bins=thresholds_brain)
		regions_brain = np.uint8(regions_brain*255)
		labeledImage = measure.label(regions_brain) 
		labeledImage = labeledImage.reshape(10,362,434)
		class345 = np.zeros(labeledImage.shape)
		class345[np.where(labeledImage==2)]=255
		class345 = np.uint8(class345)
		matterMask = self.getMask3D(class345)
		kernel = np.ones((5,5), np.uint8)
		matterMask = cv2.dilate(matterMask, kernel) 
		class012 = cv2.bitwise_not(matterMask)/255
		class012  = class012 * img
		thresholds_brain = threshold_multiotsu(class012,classes=3)
		regions_brain = np.digitize(class012, bins=thresholds_brain)
		regions_brain = np.uint8(regions_brain*255)
		ret,class1 = cv2.threshold(regions_brain,127,255,cv2.THRESH_BINARY)
		class1Inv = cv2.bitwise_not(class1)
		brainMask = self.getMask3D(class1)
		backgroundMask =  cv2.bitwise_not(brainMask)
		segmentedImg3 = matterMask.copy()
		kernel = np.ones((3,3), np.uint8)
		segmentedImg3 = cv2.erode(segmentedImg3, kernel)
		kernel = np.ones((14,14), np.uint8)
		segmentedImg3 = cv2.morphologyEx(segmentedImg3, cv2.MORPH_CLOSE, kernel)
		segmentedImg3temp = cv2.morphologyEx(matterMask, cv2.MORPH_CLOSE, kernel)
		segmentedImg3ab = np.uint8(segmentedImg3temp - segmentedImg3)
		class2 = backgroundMask - class1Inv - (segmentedImg3/255)
		class2 = np.uint8(class2*255)
		return class1,class2,segmentedImg3,segmentedImg3ab

	def custom(self):
		'''
		This funtion iteratively reads 2D gray image and stack them and does the segmentation task by calling 
		appropriate functions to process all channel simultanously and finally writes the colored image slices by slices into output path
		Arguments - 
			input path 		   : specified in init
			output path 	   : specified in init
		Returns -
			None
		'''
		print("Processing started...")
		images = []
		for file in sorted(self.files):
			if file.endswith(".png"):
				path = os.path.join(sys.argv[1],file)
				img = cv2.imread(path)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				images.append(img)
		images = np.asarray(images)
		segmentedImg1,segmentedImg2,segmentedImg3,segmentedImg3ab = self.otsuMeasureLableDilate(images)
		clustImg2 = self.segmentInnerPart(segmentedImg3,images)
		finaltemp = np.zeros(images.shape)
		finaltemp[np.where(segmentedImg3ab==255)] = 80 
		finaltemp[np.where(clustImg2==80)] = 80
		kernel = np.ones((5,5), np.uint8)
		finaltemp = cv2.morphologyEx(finaltemp, cv2.MORPH_CLOSE, kernel)
		finalimage = np.zeros(images.shape)
		finalimage[np.where(segmentedImg1==255)] = 50 
		finalimage[np.where(finaltemp==80)] = 80
		finalimage[np.where(segmentedImg2==255)] = 190 
		finalimage[np.where(clustImg2==150)] = 150
		finalimage[np.where(clustImg2==250)] = 250
		for i in range(0,10):
			img = finalimage[i].copy()
			img = cv2.merge([img,img,img]) 
			img = self.fillColor(img)
			file = str(i)+".png"
			cv2.imwrite(os.path.join(sys.argv[2],file),img)
		print("Process completed")
# Main Call
if __name__ == '__main__':
	Main()