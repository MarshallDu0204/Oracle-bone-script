import cv2
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
import extract
import os

path = extract.path

def findAdj(img,point):
	pointSet = set()
	x = point[0]
	x = int(x)
	y = point[1]
	y = int(y)
	if img[x+1][y-1][0]==0:
		pointSet.add(str(x+1)+","+str(y-1))
	if img[x+1][y+1][0]==0:
		pointSet.add(str(x+1)+","+str(y+1))
	if img[x+1][y][0]==0:
		pointSet.add(str(x+1)+","+str(y))
	if img[x-1][y-1][0]==0:
		pointSet.add(str(x-1)+","+str(y-1))
	if img[x-1][y+1][0]==0:
		pointSet.add(str(x-1)+","+str(y+1))
	if img[x-1][y][0]==0:
		pointSet.add(str(x-1)+","+str(y))
	if img[x][y-1][0]==0:
		pointSet.add(str(x)+","+str(y-1))
	if img[x][y+1][0]==0:
		pointSet.add(str(x)+","+str(y+1))

	return pointSet

def isIsolate(img,point):
	x = point[0]
	x = int(x)
	y = point[1]
	y = int(y)
	index = 0
	if img[x+1][y-1][0]==0:
		index+=1
	if img[x+1][y+1][0]==0:
		index+=1
	if img[x+1][y][0]==0:
		index+=1
	if img[x-1][y-1][0]==0:
		index+=1
	if img[x-1][y+1][0]==0:
		index+=1
	if img[x-1][y][0]==0:
		index+=1
	if img[x][y-1][0]==0:
		index+=1
	if img[x][y+1][0]==0:
		index+=1
	if index == 0:
		return True
	return False

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

finalResult = extract.getShape2()
#extract.printShape()

img = extract.outline(path)
image = extract.binaryToImg(img)



longList = []
for element in pointList:
	x = extract.getDist(image,element[0],element[1])
	#print(x)
	if x> 30 and x<100:
		longList.append(element)

for temp in longList:
	print(temp)
	point1 = temp[0]
	point2 = temp[1]
	image[point1[0]][point1[1]] = [255,255,255]
	image[point2[0]][point2[1]] = [255,255,255]
	diff1 = point2[1]-point1[1]
	diff2 = point2[0]-point1[0]
	if abs(diff1)>abs(diff2):#纵
		point11 = point1
		point12 = point1
		point21 = point2
		point22 = point2
		point11[0] = point11[0]-1
		point12[0] = point12[0]+1
		point21[0] = point21[0]-1
		point22[0] = point22[0]+1
		x1 = point11[0]
		x2 = point21[0]
		y1 = point12[1]
		y2 = point22[1]
		if x1 == x2:
			tempY = y1
			while tempY!=y2:
				image[x1][tempY] = [255,255,255]
				if y1<y2:
					tempY+=1
				else:
					tempY-=1
		else:
			k = (y1-y2)/(x1-x2)
			b = y1-k*x1
			b = int(b)
			tempX = x1
			while tempX!=x2:
				tempY = b+k*x1
				tempY = int(tempY)
				image[tempX][tempY] = [255,255,255]
				if x1<x2:
					tempX+=1
				else:
					tempX-=1

	else:#横
		point11 = point1
		point12 = point1
		point21 = point2
		point22 = point2
		point11[1] = point11[1]-1
		point12[1] = point12[1]+1
		point21[1] = point21[1]-1
		point22[1] = point22[1]+1
		x1 = point11[0]
		x2 = point21[0]
		y1 = point12[1]
		y2 = point22[1]
		if x1 == x2:
			tempY = y1
			while tempY!=y2:
				image[x1][tempY] = [255,255,255]
				if y1<y2:
					tempY+=1
				else:
					tempY-=1
		else:
			k = (y1-y2)/(x1-x2)
			b = y1-k*x1
			b = int(b)
			tempX = x1
			while tempX!=x2:
				tempY = b+k*x1
				tempY = int(tempY)
				image[tempX][tempY] = [255,255,255]
				if x1<x2:
					tempX+=1
				else:
					tempX-=1

newPointList = []
i=0
while i!=96:
	j=0
	while j!=96:
		k=0
		while k!=3:
			image[i][j][k] = int(image[i][j][k])
			k+=1
		if image[i][j][0] == 0 and image[i][j][1]==0 and image[i][j][2] ==0:
			tempList = [i,j]
			newPointList.append(tempList)
		j+=1
	i+=1


def seperateImg(image,newPointList):
	tempResult = []
	for tempIndex in newPointList:
		if not isIsolate(image,tempIndex):
			tempResult.append(tempIndex)
	newPointList = tempResult
	finalResult = []
	while len(newPointList)>5:
		point = newPointList[0]
		pointSet = findAdj(image,point)
		newSet = pointSet
		if len(pointSet)>0:
			index = 0
			while index!=1000:
				for po in newSet:
					po = po.split(",")
					newSet = newSet | findAdj(image,po)
					#print(newSet)
				index+=1
			finalResult.append(newSet)
			removeList = []
			for element in newSet:
				element = element.split(",")
				x = int(element[0])
				y = int(element[1])
				removeList.append([x,y])
			newList = []
			for element in newPointList:
				if element not in removeList:
					newList.append(element)
			newPointList = newList
			print(len(newPointList))
	return finalResult

def drawShape(image,pointList):
	finalResult = seperateImg(image,pointList)
	tempResult = []
	for element in finalResult:
		if len(element)>15:
			tempResult.append(element)
	finalResult = tempResult
	tempPair = []
	i=0
	while i!=len(tempResult):
		j=0
		while j!=len(tempResult):
			if extract.isInside(tempResult[i],tempResult[j]):
				tempValue = [i,j]
				tempPair.append(tempValue)
			j+=1
		i+=1
	print(tempPair)
	newResult = set()
	tempResult = []
	for element in tempPair:
		for index in element:
			newResult.add(index)

	i=0
	while i!=len(finalResult):
		if i not in newResult:
			tempResult.append(finalResult[i])
		i+=1

	consistList = []

	for element in tempPair:
		consistList.append(finalResult[element[0]]|finalResult[element[1]])

	finalResult = tempResult
	for element in finalResult:
		i=0
		while i!=96:
			j=0
			while j!=96:
				image[i][j] = [255,255,255]
				j+=1
			i+=1
		for attr in element:
			attr = attr.split(',')
			x = int(attr[0])
			y = int(attr[1])
			image[x][y] = [0,0,0]

		tempImage = np.array(image,dtype = 'uint8')

		tempImage = Image.fromarray(tempImage,'RGB')
		tempImage.save('predict_image.jpg')
		tempImage.show()

	for element in consistList:
		i=0
		while i!=96:
			j=0
			while j!=96:
				image[i][j] = [255,255,255]
				j+=1
			i+=1
		for attr in element:
			attr = attr.split(',')
			x = int(attr[0])
			y = int(attr[1])
			image[x][y] = [0,0,0]

		tempImage = np.array(image,dtype = 'uint8')

		tempImage = Image.fromarray(tempImage,'RGB')
		tempImage.save('predict_image.jpg')
		tempImage.show()

	

drawShape(image,newPointList)
