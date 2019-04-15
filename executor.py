import cv2
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
import extract
import os

path = extract.path

def getInfo():
	with open("C:/Users/24400/Desktop/info.txt","w") as f:
		f.write("")
 
	os.system("python extract.py")
	os.system("python extract.py")
	os.system("python extract.py")
	os.system("python extract.py")
	os.system("python extract.py")

def processInfo():

	pointList = []
	with open("C:/Users/24400/Desktop/info.txt","r") as f:
		info = f.read()
		info = info.split("-------")
		for element in info:
			element = element.split("]]")
			for x in element:
				x = x.strip()
				if len(x)!=0:
					x = x[2:len(x)]
					x = x.split("], [")
					y1 = x[0]
					y2 = x[1]
					y1 = y1.split(",")
					y2 = y2.split(",")
					point1 = [int(y1[0]),int(y1[1])]
					point2 = [int(y2[0]),int(y2[1])]
					if [point1,point2] not in pointList and [point2,point1] not in pointList:
						pointList.append([point1,point2])

	return pointList



getInfo()
pointList = processInfo()


#extract.printShape()

img = extract.outline(path)
image = extract.binaryToImg(img)


longList = []
for element in pointList:
	x = extract.getDist(image,element[0],element[1])
	print(x)
	if x> 15 and x<100:
		longList.append(element)

for element in longList:
	x = element[0]
	y = element[1]

	initX = min(x[0],y[0])
	initY = min(x[1],y[1])
	finalX = max(x[0],y[0])
	finalY = max(x[1],y[1])

	i = initX
	while i!=finalX:
		j = initY
		while j!=finalY:
			image[i][j] = [0,0,255]
			j+=1
		i+=1

'''
for temp in longList:
	for element in temp:
		image[element[0]][element[1]] = [0,0,255]
'''
tempImage = np.array(image,dtype = 'uint8')

tempImage = Image.fromarray(tempImage,'RGB')
tempImage.show()

