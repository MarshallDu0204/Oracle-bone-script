import os,shutil,stat
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
import Augmentor
import cv2
import random

def augment(img):
	i=1
	while i!=95:
		j=1
		while j!=95:
			if img[i][j]==1:
				img[i-1][j]=1
				img[i+1][j]=1
				img[i][j-1]=1
				img[i][j+1]=1
			j+=1
		i+=1


def convertToBinary(img):
	axis = []
	for x in img:
		element = []
		for y in x:
			if y[0]<150:
				element.append(1)
			else:
				element.append(0)
		element = np.array(element,dtype = 'uint8')
		axis.append(element)
	axis = np.array(axis)
	return axis


def binaryToImg(bin):
	axis = []
	for xAxis in bin:
		element = []
		for yAxis in xAxis:
			temp = []
			if yAxis == 1:
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


