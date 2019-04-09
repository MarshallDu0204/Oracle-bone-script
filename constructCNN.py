import os,shutil,stat
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
import Augmentor
import cv2
import random

train_set_writer = tf.python_io.TFRecordWriter("C:/Users/24400/Desktop/train_set_cnn.tfrecords")

def compressImg(img):
	sample_image = np.asarray(a=img[:, :, 0], dtype=np.uint8)
	return sample_image

def writeToSet():
	judgeList = [0,1]
	path1 = "C:/Users/24400/Desktop/oracle-jpg"
	path2 = "C:/Users/24400/Desktop/jin-jpg"

	temp = os.listdir(path1)
	image = []
	for img in temp:
		image.append(path1+"/"+img)

	temp = os.listdir(path2)
	label = []
	for lab in temp:
		label.append(path2+"/"+lab)

	index = 0
	while index!=420:
		trueList = []
		falseList = []
		imageList = []
		tempImg = os.listdir(image[index])
		tempLabel = os.listdir(label[index])
		trueNum = len(tempLabel)
		for element in tempImg:
			imagePath = image[index]+"/"+element
			imageList.append(imagePath)
			num = random.randint(0,trueNum-1)
			labelPath = label[index]+"/"+tempLabel[num]
			trueList.append(labelPath)
			falseNum1 = 0
			falseNum2 = 0
			if index>210:
				falseNum1 = random.randint(0,index-1)
				tempFalse = os.listdir(label[falseNum1])
				tempNum = len(tempFalse)
				falseNum2 = random.randint(0,tempNum-1)
				falsePath = label[falseNum1]+"/"+tempFalse[falseNum2]
				falseList.append(falsePath)
			else:
				falseNum1 = random.randint(index+1,419)
				tempFalse = os.listdir(label[falseNum1])
				tempNum = len(tempFalse)
				falseNum2 = random.randint(0,tempNum-1)
				falsePath = label[falseNum1]+"/"+tempFalse[falseNum2]
				falseList.append(falsePath)
		
		value = len(imageList)
		
		i = 0
		while i!=value:
			image1 = imageList[i]
			image1 = cv2.imdecode(np.fromfile(image1,dtype=np.uint8),-1)
			image1 = compressImg(image1)
			image1 = cv2.resize(src = image1,dsize=(96,96))
			image1[image1 <= 150] = 0
			image1[image1 > 150] = 255

			image2 = trueList[i]
			image2 = cv2.imdecode(np.fromfile(image2,dtype=np.uint8),-1)
			image2 = compressImg(image2)
			image2 = cv2.resize(src = image2,dsize=(96,96))
			image2[image2 <= 150] = 0
			image2[image2 > 150] = 255

			image3 = falseList[i]
			image3 = cv2.imdecode(np.fromfile(image3,dtype=np.uint8),-1)
			image3 = compressImg(image3)
			image3 = cv2.resize(src = image3,dsize=(96,96))
			image3[image3 <= 150] = 0
			image3[image3 > 150] = 255

			feature = {}
			feature['img1'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image1.tobytes()]))
			feature['img2'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image2.tobytes()]))
			feature['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[judgeList[1]]))
			example = tf.train.Example(features = tf.train.Features(feature = feature))
			train_set_writer.write(example.SerializeToString())

			feature = {}
			feature['img1'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image2.tobytes()]))
			feature['img2'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image3.tobytes()]))
			feature['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[judgeList[0]]))
			example = tf.train.Example(features = tf.train.Features(feature = feature))
			train_set_writer.write(example.SerializeToString())

			i+=1
			
		index+=1
		if index%10==0:
			print(index)
	print("Done")

writeToSet()
train_set_writer.close()
