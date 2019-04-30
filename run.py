import os
import random

def getFullList(path):
	fullList = []
	chars = os.listdir(path)
	for char in chars:
		tempPath = path+"/"+char
		imgs = os.listdir(tempPath)
		if len(imgs)>10:
			imgs = random.sample(imgs,10)
		for img in imgs:
			imgPath = tempPath+"/"+img
			fullList.append(imgPath)

	return fullList

def changePath(readPath):
	with open("extract.py","r",encoding = 'utf-8') as f:
		a = f.readlines()
		a[5] = readPath
	
	with open("extract.py","w",encoding = 'utf-8') as f:
		for element in a:
			f.writelines(element)

fullList = getFullList("/root/oracle")

start = 0
end = 1000
i=start
while i!=end:
	path = fullList[i]
	x1 = "path = '"+path+"'\n"
	changePath(x1)
	
	os.system("python3 executor.py")
	with open("result.txt","w") as f:
		f.write(str(i))
	i+=1

	

