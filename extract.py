import cv2
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt


path = "C:/Users/24400/Desktop/J13452.jpg"


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



def outline(img):
	img = cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
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


def getAdjPoint(img,point,pointList):
	x = point[0]
	y = point[1]

	pointSet = findAdj(img,[x,y])
	newSet = pointSet
	if len(pointSet)>0:
		index = 0
		resultSet = set()
		maxSet = 0
		breakIndex = 0
		while index!=1000:
			if breakIndex == 1:
				break
			for po in newSet:
				po = po.split(",")
				newSet = newSet | findAdj(img,po)
			for element in newSet:
				element = element.split(",")
				x = int(element[0])
				y = int(element[1])
				if [x,y] in pointList:
					if len(resultSet)<3:
						resultSet.add(str(x)+","+str(y))
					if len(resultSet)==3:
						breakIndex = 1
			index+=1

	adjList = []

	for element in resultSet:
		element = element.split(",")
		x = int(element[0])
		y = int(element[1])

		if [x,y]!=point:
			adjList.append([x,y])

	vec1 = [adjList[0][0]-point[0],adjList[0][1]-point[1]]
	vec2 = [adjList[1][0]-point[0],adjList[1][1]-point[1]]
	cross = vec1[0]*vec2[1]-vec1[1]*vec2[0]
	return cross
	
def getPair(path):

	img = outline(path)

	targetPoint = []

	image = binaryToImg(img)

	tempImage = []
	for x in image:
		temp = []
		for y in x:
			if y[0] == 0:
				temp.append([0,0,0])
			else:
				temp.append([255,255,255])
		tempImage.append(temp)


	gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	gray=np.float32(gray)

	dst=cv2.cornerHarris(gray,2,3,0.04) 

	image[dst>0.05*dst.max()]=[0,0,255]

	i=0
	while i!=96:
		j=0
		while j!=96:
			if image[i][j][0]==0 and image[i][j][1] == 0 and image[i][j][2]==255 and tempImage[i][j][0] == 0:
				targetPoint.append([i,j])
			j+=1
		i+=1

	concaveList = []
	for element in targetPoint:
		cross = getAdjPoint(tempImage,element,targetPoint)
		if cross >0:
			concaveList.append(element)

	closeList = []

	for element in concaveList:
		for point in concaveList:
			point1 = element
			point2 = point
			y = point1[1]-point2[1]
			x = point1[0]-point2[0]
			dis = x*x+y*y
			if dis<=100 and dis >40:
				if [point1,point2] not in closeList and [point2,point1] not in closeList:
					closeList.append([point1,point2])
	
	return closeList

def getTarget(path):
	img = outline(path)

	targetPoint = []

	image = binaryToImg(img)

	tempImage = []
	for x in image:
		temp = []
		for y in x:
			if y[0] == 0:
				temp.append([0,0,0])
			else:
				temp.append([255,255,255])
		tempImage.append(temp)


	gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	gray=np.float32(gray)

	dst=cv2.cornerHarris(gray,2,3,0.04) 

	image[dst>0.05*dst.max()]=[0,0,255]

	i=0
	while i!=96:
		j=0
		while j!=96:
			if image[i][j][0]==0 and image[i][j][1] == 0 and image[i][j][2]==255 and tempImage[i][j][0] == 0:
				targetPoint.append([i,j])
			j+=1
		i+=1

	return targetPoint

def isInside(pointList1,pointList2):
	maxAx,maxAy = 0,0
	minAx,minAy = 96,96
	maxBx,maxBy = 0,0
	minBx,minBy = 96,96

	for element in pointList1:
		element = element.split(',')
		temp1 = int(element[0])
		temp2 = int(element[1])
		if maxAx<temp1:
			maxAx = temp1
		if minAx>temp1:
			minAx = temp1
		if maxAy<temp2:
			maxAy = temp2
		if minAy>temp2:
			minAy = temp2

	for element in pointList2:
		element = element.split(',')
		temp1 = int(element[0])
		temp2 = int(element[1])
		if maxBx<temp1:
			maxBx = temp1
		if minBx>temp1:
			minBx = temp1
		if maxBy<temp2:
			maxBy = temp2
		if minBy>temp2:
			minBy = temp2

	if maxAx>maxBx and maxAy>maxBy and minAx<minBx and minAy< minBy:
		return True
	else:
		return False

def getShape(img,concaveList):
	finalResult = []

	tempList = concaveList
	while len(tempList)>1:
		point = tempList[0]
		pointSet = findAdj(img,point)
		newSet = pointSet
		if len(pointSet)>0:
			index = 0
			while index!=1000:
				for po in newSet:
					po = po.split(",")
					newSet = newSet | findAdj(img,po)

				index+=1
			finalResult.append(newSet)
			removeList = []
			for element in newSet:
				element = element.split(",")
				x = int(element[0])
				y = int(element[1])
				removeList.append([x,y])
			newList = []
			for element in tempList:
				if element not in removeList:
					newList.append(element)
			tempList = newList
	return finalResult


def printShape():
	img = outline(path)
	image = binaryToImg(img)
	info = getTarget(path)
	result = getShape(image,info)
	#print(isInside(result[0],result[2]))
	colorSet = [[0,0,255],[0,255,0],[255,0,0],[255,255,0],[0,255,255]]
	i=0
	for attr in result:
		color = colorSet[i%5]
		for element in attr:
			element = element.split(',')
			x = int(element[0])
			y = int(element[1])
			image[x][y] = color
		i+=1
	tempImage = np.array(image,dtype = 'uint8')

	tempImage = Image.fromarray(tempImage,'RGB')
	tempImage.save('predict_image.jpg')
	tempImage.show()

def getDist(img,point1,point2):

	newSet = set()
	newSet = findAdj(img,point1)
	index = 0
	while index!=200:
		for po in newSet:
			po = po.split(",")
			newSet = newSet | findAdj(img,po)
		for element in newSet:
			element = element.split(",")
			x = int(element[0])
			y = int(element[1])

			if x == point2[0] and y == point2[1]:
				return index
		index+=1
	return 0

closeList = getPair(path)
with open("C:/Users/24400/Desktop/info.txt","a") as f:
	for element in closeList:
		f.write(str(element)+"\n")
	f.write("-------\n")

