import numpy as np
from skimage.filters import threshold_multiotsu

class Threshold:
	'''
	This class is reponsible for doing MultiOTSU thresholding for 2D and 3D data
	'''
	def __init__(self,img):

		self.img = img.copy()

	def handle(self,img1,img2):
		'''
		This funnction is helping function to merge multiOTSU output to fuzzy
		Arguments - 
			img1 			   : MultiOtsu output
			img2 			   : Fuzzy output
		Returns -
			img1 	   	  	   : Merged output
		'''
		img1[np.where(img2==150)] = [150]
		img1[np.where(img2==250)] = [250]
		return img1
	
	def multiOtsu(self):
		'''
		This function is reponsible for performing multiOTSU on 2D for class 3, 4, 5
		Arguments - 
			img 			   : original grayscale image stack (region of class 345 only)
		Returns -
			new_img 	   	   : segmented class 3, 4, 5 images
		'''
		img = self.img.copy()
		thresholds_brain = threshold_multiotsu(img,classes=4)
		regions_brain = np.digitize(img, bins=thresholds_brain)
		regions_brain[np.where(regions_brain==3)] = [250]
		regions_brain[np.where(regions_brain==2)] = [150]
		regions_brain[np.where(regions_brain==1)] = [80]
		return regions_brain