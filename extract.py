import cv2
from PIL import Image
import numpy as np
import math
from shapely import wkt,geometry
import matplotlib.pyplot as plt

def compressImg(img):
	sample_image = np.asarray(a=img[:, :, 0], dtype=np.uint8)
	return sample_image

def binaryToImg(bin):
	axis = []
	for x in bin:
		element = []
		for y in x:
			temp = []
			if y == 0:
				temp.append(0)
				temp.append(0)
				temp.append(0)
			else:
				temp.append(255)
				temp.append(255)
				temp.append(255)
			temp = np.array(temp,dtype = 'uint8')
			element.append(temp)
		element = np.array(element)
		axis.append(element)

	axis = np.array(axis)
	return axis

path = "C:/Users/24400/Desktop/J17538.jpg"

def outline(img):
	img = cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
	#img = cv2.GaussianBlur(img,(3,3),0)
	#img = compressImg(img)
	img = cv2.resize(src = img,dsize=(96,96))
	canny = cv2.Canny(img,200,255)
	newImg = []
	for x in canny:
		temp = []
		for y in x:
			if y == 0:
				temp.append(255)
			if y==255:
				temp.append(0)
		newImg.append(temp)
	newImg = np.array(newImg,dtype = 'uint8')

	return newImg

img = outline(path)
#print(img)

image = binaryToImg(img)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray=np.float32(gray)
print(gray[10])
dst=cv2.cornerHarris(gray,2,3,0.04)
dst=cv2.dilate(dst,None)
print(dst[10])
image[dst>0.05*dst.max()]=[0,0,255]
#image[dst<0]=[255,255,255]

image = Image.fromarray(image,'RGB')
image.show()