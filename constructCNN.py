import os,shutil,stat
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import random

charList = []

oraclePath = "/root/oracle-jpg"
jinPath = "/root/jin-jpg"

charList = os.listdir(oraclePath)

def compressImg(img):
	sample_image = np.asarray(a=img[:, :, 0], dtype=np.uint8)
	return sample_image

def getList(index1,index2):
	char1 = charList[index1]
	char2 = charList[index2]

	oracleDir = oraclePath+"/"+str(char1)
	jinDir = jinPath+"/"+str(char2)

	sampleImg = os.listdir(oracleDir)
	targetImg = os.listdir(jinDir)

	sampleList = []
	targetList = []
	tempSample = []
	tempTarget = []

	tempSample = sampleImg
	tempTarget = targetImg

	if len(sampleImg)>5:

		tempSample = random.sample(sampleImg,5)

	if len(targetImg)>5:

		tempTarget = random.sample(targetImg,5)

	for sample in tempSample:
		for target in tempTarget:
			sampleList.append(oracleDir+"/"+sample)
			targetList.append(jinDir+"/"+target)

	return sampleList,targetList

def writeToSet():
	train_set_writer = tf.python_io.TFRecordWriter("/root/train_set_cnn.tfrecords")
	numList = []
	num = 0
	while num!=420:
		numList.append(num)
		num+=1

	i = 0
	while i!=420:
		ranList = random.sample(numList,10)
		falseList = []
		for element in ranList:
			if element is not i:
				falseList.append(element)

		sampleList = []
		targetList = []
		sampleList,targetList = getList(i,i)

		index = 0
		while index!=len(sampleList):
			image1 = cv2.imdecode(np.fromfile(sampleList[index],dtype=np.uint8),-1)
			image1 = compressImg(image1)
			image1 = cv2.resize(src = image1,dsize=(96,96))
			image2 = cv2.imdecode(np.fromfile(targetList[index],dtype=np.uint8),-1)
			image2 = compressImg(image2)
			image2 = cv2.resize(src = image2,dsize=(96,96))
			image1[image1 <= 150] = 0
			image1[image1 > 150] = 255
			image2[image2 <= 150] = 0
			image2[image2 > 150] = 255
			feature = {}
			feature['img1'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image1.tobytes()]))
			feature['img2'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image2.tobytes()]))
			feature['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[1]))
			example = tf.train.Example(features = tf.train.Features(feature = feature))
			train_set_writer.write(example.SerializeToString())
			index+=1


		for num in falseList:
			sampleList,targetList = getList(i,num)
			index = 0
			while index!=len(sampleList):
				image1 = cv2.imdecode(np.fromfile(sampleList[index],dtype=np.uint8),-1)
				image1 = compressImg(image1)
				image1 = cv2.resize(src = image1,dsize=(96,96))
				image2 = cv2.imdecode(np.fromfile(targetList[index],dtype=np.uint8),-1)
				image2 = compressImg(image2)
				image2 = cv2.resize(src = image2,dsize=(96,96))
				image1[image1 <= 150] = 0
				image1[image1 > 150] = 255
				image2[image2 <= 150] = 0
				image2[image2 > 150] = 255
				feature = {}
				feature['img1'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image1.tobytes()]))
				feature['img2'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image2.tobytes()]))
				feature['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[0]))
				example = tf.train.Example(features = tf.train.Features(feature = feature))
				train_set_writer.write(example.SerializeToString())
				index+=1

		print(i)
		i+=1
	train_set_writer.close()
	print("Done")

writeToSet()

