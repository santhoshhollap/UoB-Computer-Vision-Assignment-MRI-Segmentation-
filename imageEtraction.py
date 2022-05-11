
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import scipy.io
import os
import sys



mat = scipy.io.loadmat('Brain.mat')
img = mat['T1']
lab = mat['label']


def matlab_mat2grey(A):
  I = A
  I = cv2.normalize(A, None, 0 ,1.0 ,cv2.NORM_MINMAX,dtype=cv2.CV_32F)
  I = I * 255 
  I = np.uint8(I)
  return np.asarray(I)

for i in range(0,10):
	temp = img[:,:,i]
	temp1 = lab[:,:,i]
	new_img = matlab_mat2grey(temp)
	new_img1 = matlab_mat2grey(temp1)
	name = str(i) + ".png"
	cv2.imwrite(os.path.join(sys.argv[1],name),new_img)
	cv2.imwrite(os.path.join(sys.argv[2],name),new_img1)
