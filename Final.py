'''
File			: Final.py
Application		: 2D and 3D MRI segmentation 
Author			: Santhosh Holla Prakash
Creation Date	           : 13/05/2022
description		: This .py file will read the input MRI images (which is normalized to 0-255) and process it to provide the segmented image. 
Usage			: python3 Finay.py <input_path> <output_path>
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
	This class is responsible for processing the 2D and 3D MRI image,based on the method that is configured in constructor and output the segmented image
	'''
	def __init__(self): 
		'''
		Constructor to initialize the input path, configure the method to be experimented
		'''
		self.files = sorted(os.listdir(sys.argv[1]))
		self.class12Extraction = "multiotsu_measureLable_dilate" #For stage 1-can specify chanvese_contour, but due to space issues removed. Refer git for full.
		self.matterMethod = "fuzzymultiotsu"  #can specify fuzzy or multiotsu for stage 2
		self.type= "2D" #2D or 3D data processing
		if self.type == "2D":
			self.custom2D()
		else:
			self.custom3D()

	def getMask2D(self,img):
		'''
		Function to does the operation similar to opencv drawcontour, it with specified color
		Arguments -
			img      : A image with a closed boundary
		Returns -
			im_out   : A image with a filled closed boundary with specified color
		'''
		im_floodfill = img.copy()
		h, w = img.shape[:2]
		mask = np.zeros((h+2, w+2), np.uint8)
		cv2.floodFill(im_floodfill, mask, (0,0), 255)
		im_floodfill_inv = cv2.bitwise_not(im_floodfill)
		im_out = (img | im_floodfill_inv)
		return np.asarray(im_out)

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
		This function performs the segmentation of inner matter mask as per the configured method in init (due to space issue removed other methods, refer git)
		Arguments -
			segmentedImg3      : Mask of the inner matter - inner brain region 
			img 	        : original grayscale image / image stack if 3D
		Returns -
			clustImg           : Inner brain region segmented images 
		'''
		if self.matterMethod == "fuzzymultiotsu":
			clust1 = fuzzyCluster(segmentedImg3*img,4)
			if self.type == "2D":
				clustImg1 = clust1.cluster()
			else:
				clustImg1 = clust1.cluster3D()
			clust2 = Threshold(np.uint8(segmentedImg3*img)*255)
			clustImg2 = clust2.multiOtsu()
			clustImg = clust2.handle(clustImg2,clustImg1)
			return clustImg

	def otsuMeasureLableDilate(self,img):
		'''
		This function separate the inner matter from the outer ring using multi otsu method and measure label connective detection method and segments class 012
		Arguments - 
			img 	              : original grayscale image
		Returns -
			segmentedImg1 	   : class 1 mask
			segmentedImg2 	   : class 2 mask
			segmentedImg3 	   : inner matter mask - class345
			segmentedImg3ab          : For boosting accuracy of class 3, extra periphery region mask
		'''
		thresholds_brain = threshold_multiotsu(img,classes=2)
		regions_brain = np.digitize(img, bins=thresholds_brain)
		regions_brain = np.uint8(regions_brain*255)
		labeledImage = measure.label(regions_brain) 
		class345 = np.zeros(labeledImage.shape)
		class345[np.where(labeledImage==2)]=255
		class345 = np.uint8(class345)
		matterMask = self.getMask2D(class345)
		kernel = np.ones((5,5), np.uint8)
		matterMask = cv2.dilate(matterMask, kernel) 
		class012 = cv2.bitwise_not(matterMask)/255
		class012  = class012 * img
		thresholds_brain = threshold_multiotsu(class012,classes=3)
		regions_brain = np.digitize(class012, bins=thresholds_brain)
		regions_brain = np.uint8(regions_brain*255)
		ret,class1 = cv2.threshold(regions_brain,127,255,cv2.THRESH_BINARY)
		class1Inv = cv2.bitwise_not(class1)
		brainMask = self.getMask2D(class1)
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

	def otsuMeasureLableDilate3D(self,img):
		'''
		This function separate the inner matter from the outer ring using skimage's multiotsu method and skimage's
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

	def custom2D(self):
		'''
		This function reads 2D gray image and segments by calling appropriate functions and finally writes the segmented image into output path
		Arguments - 
			input path               : specified in init
			output path 	   : specified in init
		Returns -
			None
		'''
		for file in self.files:
			if file.endswith(".png"):
				path = os.path.join(sys.argv[1],file)
				img = cv2.imread(path)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				segmentedImg1,segmentedImg2,segmentedImg3,segmentedImg3ab = self.otsuMeasureLableDilate(img)
				clustImg2 = self.segmentInnerPart(segmentedImg3,img)
				finaltemp = np.zeros(img.shape)
				finaltemp[np.where(segmentedImg3ab==255)] = 80 
				finaltemp[np.where(clustImg2==80)] = 80
				kernel = np.ones((3,3), np.uint8)
				finaltemp = cv2.morphologyEx(finaltemp, cv2.MORPH_CLOSE, kernel)
				finalimage = np.zeros(img.shape)
				finalimage[np.where(segmentedImg1==255)] = 50 
				finalimage[np.where(segmentedImg2==255)] = 190 
				finalimage[np.where(finaltemp==80)] = 80
				finalimage[np.where(clustImg2==150)] = 150
				finalimage[np.where(clustImg2==250)] = 250
				cv2.imwrite(os.path.join(sys.argv[2],file),finalimage)

	def custom3D(self):
		'''
		This funtion iteratively reads 2D gray image and stack them and does the segmentation task by calling 
		appropriate functions to process all channel simultanously and finally writes the image slices by slices into output path
		Arguments - 
			input path 		   : specified in init
			output path 	   : specified in init
		Returns -
			None
		'''
		images = []
		for file in sorted(self.files):
			if file.endswith(".png"):
				path = os.path.join(sys.argv[1],file)
				img = cv2.imread(path)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				images.append(img)
		images = np.asarray(images)
		segmentedImg1,segmentedImg2,segmentedImg3,segmentedImg3ab = self.otsuMeasureLableDilate3D(images)
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
			file = str(i)+".png"
			cv2.imwrite(os.path.join(sys.argv[2],file),img)
# Main Call
if __name__ == '__main__':
	Main()