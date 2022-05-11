import numpy as np
import cv2
import skfuzzy as fuzz
import os
import sys

class fuzzyCluster:
	'''This class is reponsible for clustering the image using fuzzy algorithm'''
	def __init__(self,img,nCluster):
		self.img = img.copy()
		self.numCluster = nCluster

	def getRemain(self,valmax,valmin):
		'''
		This function is supporting function for just coloring.
		'''
		colors = [0,80,150,250]
		for color in colors:
			if color not in [valmax,valmin]:
				remain1 = color
				break
		for color in colors:
			if color not in [valmax,valmin,remain1]:
				remain2 = color
				break
		return remain1,remain2

	def getMinMax(self,new_img):
		'''
		This function is supporting function for just coloring.
		'''
		classes= []
		colors = [0,80,150,250]
		for i in range(0,4):
			classes.append(len(np.where(new_img==colors[i])[0]))
		mini = min(classes)
		maxi = max(classes)
		for i in range(0,4):
			if mini == classes[i]:
				valmin = colors[i]
				break
		for i in range(0,4):
			if maxi == classes[i]:
				valmax = colors[i]
				break
		return valmin,valmax

	def cluster(self):
		'''
		This function is reponsible for performing fuzzy clustering algorithm on 2D for class 3, 4, 5
		Arguments - 
			img 			   : original grayscale image (region of class 345 only)
		Returns -
			new_img 	   	   : segmented class 3, 4, 5 image
		'''
		dim = self.img.shape
		self.img = self.img.reshape(1,self.img.shape[0]*self.img.shape[1])
		cntr, u, u0, d, jm, p, fpc =fuzz.cluster.cmeans(self.img,self.numCluster,2,0.5,20)
		new_img= self.changeColorFuzzycmeans(u,cntr)
		new_img = new_img.reshape(dim)
		valmin,valmax = self.getMinMax(new_img)
		new_img1 = np.zeros(new_img.shape)
		new_img2 = np.zeros(new_img.shape)
		new_img3 = np.zeros(new_img.shape)
		new_img4 = np.zeros(new_img.shape)
		new_img1[np.where(new_img==valmin)] = [255]
		kernel = np.ones((5,5),np.uint8)
		new_img1 = cv2.morphologyEx(new_img1, cv2.MORPH_CLOSE, kernel)
		new_img1[np.where(new_img1==255)] = [80]
		remain1, remain2 = self.getRemain(valmax,valmin)
		new_img3[np.where(new_img==remain1)] = [250]
		new_img4[np.where(new_img==remain2)] = [250]
		new_img3 = np.uint8(new_img3)
		new_img4 = np.uint8(new_img4)
		contours, hierarchy = cv2.findContours(new_img3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		cnt = sorted(contours, key=cv2.contourArea)
		x,y,w,h = cv2.boundingRect(cnt[-1])
		area1 = w * h
		contours, hierarchy = cv2.findContours(new_img4, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		cnt = sorted(contours, key=cv2.contourArea)
		x,y,w,h = cv2.boundingRect(cnt[-1])
		area2 = w * h
		if area1 > area2:
			new_img3[np.where(new_img==remain1)] = 150
		else:
			new_img4[np.where(new_img==remain2)] = 150
		new_img = new_img1 + new_img2 + new_img3 + new_img4
		new_img[np.where(new_img==230)] = [150]
		return new_img

	def cluster3D(self):
		'''
		This function is reponsible for performing fuzzy clustering algorithm on 3D for class 3, 4, 5
		Arguments - 
			img 			   : original grayscale image stack (region of class 345 only)
		Returns -
			new_img 	   	   : segmented class 3, 4, 5 image
		'''
		dim = self.img.shape
		self.img  = self.img.reshape(1,362*434*10)
		cntr, u, u0, d, jm, p, fpc =fuzz.cluster.cmeans(self.img ,self.numCluster,2,0.05,300)
		new_img= self.changeColorFuzzycmeans(u,cntr)
		newImg = new_img.reshape(dim)
		newImgs = []
		for i in range(0,10):
			new_img = newImg[i].copy()
			valmin,valmax = self.getMinMax(new_img)
			new_img1 = np.zeros(new_img.shape)
			new_img2 = np.zeros(new_img.shape)
			new_img3 = np.zeros(new_img.shape)
			new_img4 = np.zeros(new_img.shape)
			new_img1[np.where(new_img==valmin)] = [255]
			kernel = np.ones((5,5),np.uint8)
			new_img1 = cv2.morphologyEx(new_img1, cv2.MORPH_CLOSE, kernel)
			new_img1[np.where(new_img1==255)] = [80]
			remain1, remain2 = self.getRemain(valmax,valmin)
			new_img3[np.where(new_img==remain1)] = [250]
			new_img4[np.where(new_img==remain2)] = [250]
			new_img3 = np.uint8(new_img3)
			new_img4 = np.uint8(new_img4)
			contours, hierarchy = cv2.findContours(new_img3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
			cnt = sorted(contours, key=cv2.contourArea)
			x,y,w,h = cv2.boundingRect(cnt[-1])
			area1 = w * h
			contours, hierarchy = cv2.findContours(new_img4, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
			cnt = sorted(contours, key=cv2.contourArea)
			x,y,w,h = cv2.boundingRect(cnt[-1])
			area2 = w * h
			if area1 > area2:
				new_img3[np.where(new_img==remain1)] = 150
			else:
				new_img4[np.where(new_img==remain2)] = 150
			new_img = new_img1 + new_img2 + new_img3 + new_img4
			new_img[np.where(new_img==230)] = [150]
			newImgs.append(new_img) 
		return np.asarray(newImgs)
	
	def changeColorFuzzycmeans(self,cluster_membership, clusters):
		'''
		This function is supporting function for just coloring.
		'''
		color1 = [0,      80,     150,     250] #R
		img1 = []
		clusters = list(clusters)
		clusters.sort()
		for pix in cluster_membership.T:
			if clusters[np.argmax(pix)] == clusters[0]:
				img1.append(color1[0])
			elif clusters[np.argmax(pix)] == clusters[1]:
				img1.append(color1[1])
			elif clusters[np.argmax(pix)] == clusters[2]:
				img1.append(color1[2])
			elif clusters[np.argmax(pix)] == clusters[3]:
				img1.append(color1[3])
		return np.asarray(img1)