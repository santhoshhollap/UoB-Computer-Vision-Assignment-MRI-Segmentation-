import numpy as np
import cv2
import os
import sys
from skimage.metrics import structural_similarity


files = os.listdir(sys.argv[2])

def getMasksforGT(img):
	'''
	This function is responsible for generating the mask for groundtruth images
	Input : 3 grayimages
	output : masked images
	'''
	image = []
	colors = [0,51,102,153,204,255]
	for i in range(0,6):
		tempImg = np.zeros(img.shape)
		tempImg[np.where(img==colors[i])] = 255
		image.append(tempImg)
	return image

def getMasksforPTgray(img):
	'''
	This function is responsible for generating the mask for segmented 2D images
	Input : grayimages
	output : masked images
	'''
	image = []
	colors = [0,50,190,80,150,250]
	for i in range(0,6):
		tempImg = np.zeros(img.shape)
		tempImg[np.where(img==colors[i])] = 255
		image.append(tempImg)
	return image

def getMasksforPT(img):
	'''
	This function is responsible for generating the mask for segmented 2D images which was colored
	Input : grayimages
	output : masked images
	'''
	image = []
	colors = [16,98,184,226,111,38]
	for i in range(0,6):
		tempImg = np.zeros(img.shape)
		tempImg[np.where(img==colors[i])] = 255
		image.append(tempImg)
	return image

def calculateIOU(target,prediction):
	'''
	This function is responsible for calculating the IOU for given masks
	Input : Segmented mask, groundtruth mask
	output : IOU value
	'''
	intersection = np.logical_and(target, prediction)
	union = np.logical_or(target, prediction)
	iou_score = np.sum(intersection) / np.sum(union)
	return iou_score

class0 = []
class1 = []
class2 = []
class3 = []
class4 = []
class5 = []
class0s = []
class1s = []
class2s = []
class3s = []
class4s = []
class5s = []
colored = False #if passing colored images, then flag has to be true
for file in files:
	if file.endswith(".png"):
		print(file)
		groundTruthImg = cv2.imread(os.path.join(sys.argv[1],file),0)
		predictImg = cv2.imread(os.path.join(sys.argv[2],file))
		predictImg = cv2.cvtColor(predictImg, cv2.COLOR_BGR2GRAY)
		Gtimgs = getMasksforGT(groundTruthImg)
		if colored:
			Pimgs = getMasksforPT(predictImg)
		else:
			Pimgs = getMasksforPTgray(predictImg)
		IOUval = []
		score = []
		for i in range(0,6):
			val = calculateIOU(Gtimgs[i],Pimgs[i])
			(scores, diff) = structural_similarity(Gtimgs[i], Pimgs[i], full=True)
			score.append(scores)
			IOUval.append(val)
		print(IOUval)
		print(score)

		class0.append(IOUval[0])
		class1.append(IOUval[1])
		class2.append(IOUval[2])
		class3.append(IOUval[3])
		class4.append(IOUval[4])
		class5.append(IOUval[5])
		class0s.append(score[0])
		class1s.append(score[1])
		class2s.append(score[2])
		class3s.append(score[3])
		class4s.append(score[4])
		class5s.append(score[5])
lis = []
lis.append(np.average(class0))
lis.append(np.average(class1))
lis.append(np.average(class2))
lis.append(np.average(class3))
lis.append(np.average(class4))
lis.append(np.average(class5))
print("\nAverage IOU score ", lis)
print("\n")
lis = []
lis.append(np.average(class0s))
lis.append(np.average(class1s))
lis.append(np.average(class2s))
lis.append(np.average(class3s))
lis.append(np.average(class4s))
lis.append(np.average(class5s))
print("\nAverage SSIM score ", lis)
print("\n")