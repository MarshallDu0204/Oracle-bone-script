import cv2
import PIL
import numpy as np
import math
from shapely import wkt,geometry
import matplotlib.pyplot as plt

def binaryToImg(bin):
	axis = []
	for x in bin:
		element = []
		for y in x:
			temp = []
			if y == 1:
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
	img = cv2.resize(src = img,dsize=(96,96))
	canny = cv2.Canny(img, 50, 150)
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

def removeSingleAdj(img):

	img = img

	i=1
	while i!=95:
		j=1
		
		while j!=95:
			adjNum = 0
			if img[i][j]==0:
				if img[i][j+1]==0:
					adjNum+=1
				if img[i][j-1]==0:
					adjNum+=1
				if img[i-1][j]==0:
					adjNum+=1
				if img[i+1][j]==0:
					adjNum+=1
				if adjNum!=2:
					img[i][j]==255
			j+=1
		i+=1

	return img



img = outline(path)
img = removeSingleAdj(img)

def giveCoord(img):
	coordList = []
	i=0
	while i!=96:
		j=0
		while j!=96:
			if img[i][j]==0:
				tempstr = str(i)+" "+str(j)
				coordList.append(tempstr)
			j+=1
		i+=1
	return coordList

coordList = giveCoord(img)
print(coordList)
		


class Point:
	"""点类"""
	x=0.0
	y=0.0
	index=0 #点在线上的索引

	def __init__(self,x,y,index):
		self.x=x
		self.y=y
		self.index=index

class Douglas:
	points=[]
	D=1 #容差
 
	def readPoint(self,text):
		"""生成点要素"""
		g=wkt.loads(text)
		coords=g.coords
		for i in range(len(coords)):
			self.points.append(Point(coords[i][0],coords[i][1],i))
 
	def compress(self,p1,p2):
		"""具体的抽稀算法"""
		swichvalue=False
		#一般式直线方程系数 A*x+B*y+C=0,利用点斜式
		#A=(p1.y-p2.y)/math.sqrt(math.pow(p1.y-p2.y,2)+math.pow(p1.x-p2.x,2))
		A=(p1.y-p2.y)
		#B=(p2.x-p1.x)/math.sqrt(math.pow(p1.y-p2.y,2)+math.pow(p1.x-p2.x,2))
		B=(p2.x-p1.x)
		#C=(p1.x*p2.y-p2.x*p1.y)/math.sqrt(math.pow(p1.y-p2.y,2)+math.pow(p1.x-p2.x,2))
		C=(p1.x*p2.y-p2.x*p1.y)
		
		m=self.points.index(p1)
		n=self.points.index(p2)
		distance=[]
		middle=None
 
		if(n==m+1):
			return
		#计算中间点到直线的距离
		for i in range(m+1,n):
			d=abs(A*self.points[i].x+B*self.points[i].y+C)/math.sqrt(math.pow(A,2)+math.pow(B,2))
			distance.append(d)
 
		dmax=max(distance)
 
		if dmax>self.D:
			swichvalue=True
		else:
			swichvalue=False
 
		if(not swichvalue):
			for i in range(m+1,n):
				del self.points[i]
		else:
			for i in range(m+1,n):
				if(abs(A*self.points[i].x+B*self.points[i].y+C)/math.sqrt(math.pow(A,2)+math.pow(B,2))==dmax):
					middle=self.points[i]
			self.compress(p1,middle)
			self.compress(middle,p2)
 
	def printPoint(self):
		for p in self.points:
			print(p)
 
def main():
	text = "LINESTRING(4 46,5 43,5 44,5 45,5 47)"
	'''
	text = "LINESTRING("
	for element in range(5):
		text = text+str(element)+","
	text = text[0:len(text)-1]
	text = text+")"
	'''
	#print(text)
	d=Douglas()
	d.readPoint(text)
	#d.printPoint()
	#结果图形的绘制，抽稀之前绘制
	fig=plt.figure()
	a1=fig.add_subplot(121)
	dx=[]
	dy=[]
	for i in range(len(d.points)):
		dx.append(d.points[i].x)
		dy.append(d.points[i].y)
	a1.plot(dx,dy,color='g',linestyle='-',marker='+')
 
	
	d.compress(d.points[0],d.points[len(d.points)-1])
 
	#抽稀之后绘制
	dx1=[]
	dy1=[]
	a2=fig.add_subplot(122)
	for p in d.points:
		dx1.append(p.x)
		dy1.append(p.y)
	a2.plot(dx1,dy1,color='r',linestyle='-',marker='+')
 
	#print "========================\n"
	#d.printPoint()
 
	plt.show()

main()