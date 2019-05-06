import cv2
import numpy as np
from PIL import Image
import os

def compressImg(img):
	sample_image = np.asarray(a=img[:, :, 0], dtype=np.uint8)
	return sample_image
	

def readImg(path):
	image = cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
	image = cv2.resize(src = image,dsize=(96,96))
	image = compressImg(image)
	pointList = []
	i=0
	while i!=96:
		j=0
		while j!=96:
			if image[i][j]<=200:
				temp = [i,j]
				pointList.append(temp)
			j+=1
		i+=1

	x = 0
	y = 0

	for element in pointList:
		x+=element[0]
		y+=element[1]
	x = x/len(pointList)
	y = y/len(pointList)

	x = int(x)
	y = int(y)

	return x,y,pointList

def movePoint(x,y,pointList):
	moveX = 48-x
	moveY = 48-y
	for element in pointList:
		element[0] = element[0]+moveX
		element[1] = element[1]+moveY
	image = []
	i=0
	while i!=96:
		tempArray = []
		j=0
		while j!=96:
			temp = [255,255,255]
			tempArray.append(temp)
			j+=1
		image.append(tempArray)
		i+=1

	for element in pointList:

		tempX = element[0]
		tempY = element[1]

		if tempX>=0 and tempX<=96 and tempY>=0 and tempY<=96:
			image[tempX][tempY] = [0,0,0]

	image = np.asarray(image,dtype = 'uint8')
	image = Image.fromarray(image,'RGB')
	return image



def processImg():
	path = "C:/Users/24400/Desktop/multipleCompose/"
	destPath = "C:/Users/24400/Desktop/multiple/"
	paths = os.listdir(path)
	i=0
	for element in paths:
		tempPath = path+element
		x,y,pointList = readImg(tempPath)
		try:
			image = movePoint(x,y,pointList)
			image.save(destPath+str(i)+".jpg")
			i+=1
		except:
			print("error")
		print(i)

processImg()


