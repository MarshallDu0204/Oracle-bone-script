import tensorflow as tf
import numpy as np
import cv2
from PIL import Image as Ige
import random
import os

def writeInfo(text):
	with open("info1.txt","a") as f:
		f.write(str(text)+"\n")


def compressImg(img):
	if len(img[0][0])==3:
		sample_image = np.asarray(a=img[:, :, 0], dtype=np.uint8)
		return sample_image
	if len(img[0][0])==4:
		newImg = []
		i=0
		while i!=96:
			tempList = []
			j=0
			while j!=96:
				tempList.append(255-img[i][j][3])
				j+=1
			newImg.append(tempList)
			i+=1
		img = np.asarray(a = newImg,dtype = np.uint8)
		return img


def readData_single(path):
	path = path+"/train_set_cnn.tfrecords"

	filename_queue = tf.train.string_input_producer([path],num_epochs = 20,shuffle = True)

	reader = tf.TFRecordReader()

	_,serialized_example = reader.read(filename_queue)

	features = tf.parse_single_example(serialized_example,features = {
			'img1': tf.FixedLenFeature([], tf.string),
			'img2': tf.FixedLenFeature([], tf.string),
			'label': tf.FixedLenFeature([], tf.int64)
		})

	image1 = tf.decode_raw(features['img1'],tf.uint8)

	image1 = tf.reshape(image1,[96,96,1])

	image1 = tf.cast(image1, tf.float32) * (1. / 255) - 0.5

	image2 = tf.decode_raw(features['img2'],tf.uint8)

	image2 = tf.reshape(image2,[96,96,1])

	image2 = tf.cast(image2, tf.float32) * (1. / 255) - 0.5

	label = tf.cast(features['label'], tf.int32)

	return image1,image2,label



class CNN:
	def __init__(self):

		self.input_image1 = None

		self.input_image2 = None

		self.label = None

		self.input_image1 = None

		self.input_image2 = None

		self.input_label = None

		self.keep_prob = tf.placeholder(dtype=tf.float32)

		self.lamb = tf.placeholder(dtype = tf.float32)

		self.prediction = None

		self.correct_prediction = None

		self.accurancy = None

		self.loss = None

		self.train_step = None


	def one_hot(self,labels,Label_class):
		one_hot_label = np.array([[int(i == int(labels[j])) for i in range(Label_class)] for j in range(len(labels))])   
		return one_hot_label

		
	def weight_variable(self,shape):
	    initial = tf.truncated_normal(shape,stddev=tf.sqrt(x = 2/(shape[0]*shape[1]*shape[2])))
	    tf.add_to_collection(name = 'loss',value=tf.contrib.layers.l2_regularizer(self.lamb)(initial))   
	    return tf.Variable(initial)


	def weight_variable_alter(self,shape):
	    initial = tf.truncated_normal(shape,stddev=0.2)
	    tf.add_to_collection(name = 'loss',value=tf.contrib.layers.l2_regularizer(self.lamb)(initial))   
	    return tf.Variable(initial)


	def bias_variable(self,shape):
	    initial = tf.constant(0.0,shape=shape)
	    return tf.Variable(initial_value = initial)


	def conv2d(self,x,W):
	    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding = 'SAME')

	def max_pooling(self,x):
	    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

	def merge_img(self,img1,img2):
		return tf.concat(values = [img1,img2],axis = 1)


	def setup_network(self,batch_size):

		self.input_image1 = tf.placeholder(dtype = tf.float32,shape = [batch_size,96,96,1])

		self.input_image2 = tf.placeholder(dtype = tf.float32,shape = [batch_size,96,96,1])

		self.input_label = tf.placeholder(dtype = tf.float32,shape = [batch_size,2])

		#2*96*96*1 --> 2*48*48*32

		with tf.name_scope('first_convolution'):


			#----- convolution -------

			w_conv1 = self.weight_variable([3,3,1,32])
			b_conv1 = self.bias_variable([32])

			X1 = tf.nn.relu(self.conv2d(self.input_image1,w_conv1)+b_conv1)

			w_conv2 = self.weight_variable([3,3,1,32])
			b_conv2 = self.bias_variable([32])

			X2 = tf.nn.relu(self.conv2d(self.input_image2,w_conv2)+b_conv2)

			#----- maxpooling ---------

			X1 = self.max_pooling(X1)

			X2 = self.max_pooling(X2)

		#2*48*48*32 --> 2*24*24*64

		with tf.name_scope('second_convolution'):

			#------- convolution --------

			w_conv1 = self.weight_variable([3,3,32,64])
			b_conv1 = self.bias_variable([64])

			X1 = tf.nn.relu(self.conv2d(X1,w_conv1)+b_conv1)

			w_conv2 = self.weight_variable([3,3,32,64])
			b_conv2 = self.bias_variable([64])

			X2 = tf.nn.relu(self.conv2d(X2,w_conv2)+b_conv2)

			#-------maxpooling----------

			X1 = self.max_pooling(X1)

			X2 = self.max_pooling(X2)

		#2*24*24*64 --> 2*12*12*128

		with tf.name_scope('third_convolution'):

			#-------- convolution -------

			w_conv1 = self.weight_variable([3,3,64,128])
			b_conv1 = self.bias_variable([128])

			X1 = tf.nn.relu(self.conv2d(X1,w_conv1)+b_conv1)

			w_conv2 = self.weight_variable([3,3,64,128])
			b_conv2 = self.bias_variable([128])

			X2 = tf.nn.relu(self.conv2d(X2,w_conv2)+b_conv2)

			#-------maxpooling----------

			X1 = self.max_pooling(X1)

			X2 = self.max_pooling(X2)

		#2*12*12*128 --> 2*6*6*256

		with tf.name_scope('fourth_convolution'):

			#-------- convolution -------

			w_conv1 = self.weight_variable([3,3,128,256])
			b_conv1 = self.bias_variable([256])

			X1 = tf.nn.relu(self.conv2d(X1,w_conv1)+b_conv1)

			w_conv2 = self.weight_variable([3,3,128,256])
			b_conv2 = self.bias_variable([256])

			X2 = tf.nn.relu(self.conv2d(X2,w_conv2)+b_conv2)

			#-------maxpooling----------

			X1 = self.max_pooling(X1)

			X2 = self.max_pooling(X2)

			
		#hidden layer 1  2*6*6*256 --> 1024 -->2048

		with tf.name_scope('hidden_layer_1'):

			#-------reshape---------

			X1 = tf.reshape(X1,[batch_size,6*6*256])

			X2 = tf.reshape(X2,[batch_size,6*6*256])

			X = self.merge_img(X1,X2)

			w_conv = self.weight_variable_alter([2*6*6*256,1024])
			b_conv = self.bias_variable([1024])

			X = tf.nn.relu(tf.matmul(X,w_conv)+b_conv)

			X = tf.nn.dropout(X,keep_prob = self.keep_prob)

		#final layer  2048 -- > 2

		with tf.name_scope('final_layer'):

			w_conv = self.weight_variable_alter([1024,2])
			b_conv = self.bias_variable([2])

			X = tf.matmul(X,w_conv)+b_conv

			self.prediction = X


		#softmax loss

		with tf.name_scope('softmax'):

			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.input_label))

	    #accurancy
		with tf.name_scope('accurancy'):
			
			self.correct_prediction = tf.equal(tf.argmax(self.prediction,1),tf.argmax(self.input_label,1)) 

			self.accurancy = tf.reduce_mean(tf.cast(self.correct_prediction,tf.float32)) 

		#optimize

		with tf.name_scope('gradient_descent'):

			self.train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)



	def train(self,batch_size,path):

		ckpt_path = path+"/ckpt-cnn-alter/model.ckpt"
		
		tf.summary.scalar("loss", self.loss)
			
		tf.summary.scalar('accuracy', self.accurancy)
			
		merged_summary = tf.summary.merge_all()

		model_dir = path+"/data/model"

		tb_dir = path+"/data/logs"

		all_parameters_saver = tf.train.Saver()


		with tf.Session() as sess:

			image1,image2,label = readData_single(path)

			image_batch1,image_batch2,label_batch = tf.train.shuffle_batch([image1,image2,label],batch_size = batch_size,num_threads = 4,capacity = 1012,min_after_dequeue = 1000)


			sess.run(tf.global_variables_initializer())
				
			sess.run(tf.local_variables_initializer())

			summary_writer = tf.summary.FileWriter(tb_dir, sess.graph)
				
			tf.summary.FileWriter(model_dir, sess.graph)
				
			coord = tf.train.Coordinator()
				
			threads = tf.train.start_queue_runners(coord = coord)

			try:

				epoch = 1

				while not coord.should_stop():


					example1,example2,label = sess.run([image_batch1,image_batch2,label_batch])

					label = self.one_hot(label,2)


					lo,acc,summary = sess.run([self.loss,self.accurancy,merged_summary],feed_dict = {
							self.input_image1:example1,self.input_image2:example2,self.input_label:label,self.keep_prob:1.0,self.lamb:0.004
						})

					summary_writer.add_summary(summary, epoch)

					sess.run([self.train_step],feed_dict={
							self.input_image1: example1,self.input_image2:example2,self.input_label:label,self.keep_prob: 0.5,
							self.lamb: 0.004
						})

					epoch+=1

					if epoch%10 == 0:
						writeInfo(str(epoch)+" "+str(lo)+" "+str(acc))
						print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))

					if epoch%300 == 0:
						all_parameters_saver.save(sess = sess,save_path = ckpt_path)

			except tf.errors.OutOfRangeError:

				print('Done training -- epoch limit reached')	


			finally:

				all_parameters_saver.save(sess = sess,save_path = ckpt_path)
				coord.request_stop()

			coord.join(threads)

			print("done training")


	def estimate(self,batch_size,path):

		imgPath1 = path+"/J17522.jpg"

		img1 = cv2.imdecode(np.fromfile(imgPath1,dtype=np.uint8),-1)
		img1 = cv2.resize(src = img1,dsize=(96,96))
		img1 = compressImg(img1)
		

		data1 = img1
			

		newImg1 = []

		i=0
		while i!=batch_size:
			newImg1.append(data1)
			i+=1

		data1 = np.reshape(a=newImg1, newshape=(batch_size,96,96,1))

		imgPath2 = path+"/J17538.jpg"

		img2 = cv2.imdecode(np.fromfile(imgPath2,dtype=np.uint8),-1)
		img2 = cv2.resize(src = img2,dsize=(96,96))
		img2 = compressImg(img2)
		

		data2 = img2

		newImg2 = []

		i=0
		while i!=batch_size:
			newImg2.append(data2)
			i+=1
			
		data2 = np.reshape(a=newImg2, newshape=(batch_size,96,96,1))



		ckpt_path = path+"/ckpt-cnn-alter/model.ckpt"

		all_parameters_saver = tf.train.Saver()

		with tf.Session() as sess:  
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())
			all_parameters_saver.restore(sess=sess, save_path=ckpt_path)
			predict_result = sess.run(
								tf.argmax(input=self.prediction, axis=1), 
								feed_dict={
									self.input_image1: data1,self.input_image2:data2,
									self.keep_prob: 1.0, self.lamb: 0.004
								}
							)

			predict_result = predict_result[0]
				
			print(predict_result) 
		print('Done prediction')

	def deepEstimate(self,basePath,batch_size,index1,index2):

		oraclePath = basePath+"/testOracle"

		jinPath = basePath+"/testJin"

		oracleList = os.listdir(oraclePath)

		jinList = os.listdir(jinPath)

		oraclePath = oraclePath+"/"+oracleList[index1]

		jinPath = jinPath+"/"+jinList[index2]

		oracleList = os.listdir(oraclePath)

		jinList = os.listdir(jinPath)

		ckpt_path = basePath+"/ckpt-cnn-alter/model.ckpt"

		all_parameters_saver = tf.train.Saver()

		totalNum = 0
		trueNum = 0
		falseNum = 0

		with tf.Session() as sess:  
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())
			all_parameters_saver.restore(sess=sess, save_path=ckpt_path)

			for element in oracleList:

				elementPath = oraclePath+"/"+element

				for char in jinList:

					targetPath = jinPath+"/"+char

					print(elementPath,targetPath)

					originImg = cv2.imdecode(np.fromfile(elementPath,dtype=np.uint8),-1)
					originImg = cv2.resize(src = originImg,dsize=(96,96))
					originImg = compressImg(originImg)
					originImg = originImg/255 - 0.5

					targetImg = cv2.imdecode(np.fromfile(targetPath,dtype=np.uint8),-1)
					targetImg = cv2.resize(src = targetImg,dsize=(96,96))
					targetImg = compressImg(targetImg)
					targetImg = targetImg/255 - 0.5

					origin = []
					target = []

					i=0
					
					while i!=batch_size:
						origin.append(originImg)
						target.append(targetImg)
						i+=1

					data1 = np.reshape(a=origin, newshape=(batch_size,96,96,1))
					data2 = np.reshape(a=target, newshape=(batch_size,96,96,1))


					predict_result = sess.run(
							tf.argmax(input=self.prediction, axis=1), 
											feed_dict={
												self.input_image1:data1,self.input_image2:data2,
												self.keep_prob: 1.0, self.lamb: 0.004
											}
										)

					predict_result = predict_result[0]

					totalNum+=1

			
							
					#print(predict_result)

					with open("solution.txt","a") as f:
						f.write(str(index1)+" "+str(index2)+" "+str(predict_result)+"\n")


		print('Done prediction')
		

def main():
	basePath = "/root"
	cnn = CNN()
	cnn.setup_network(64)
	
	cnn.train(64,basePath)
	#cnn.estimate(64,basePath)
	#cnn.deepEstimate(basePath,64,5,5)
	'''
	numList = [360, 67, 495, 124, 314, 669, 141, 339, 587, 334, 193, 116, 607, 211, 417, 602, 511, 8, 83, 204]
	for num in numList:
		cnn.deepEstimate(basePath,64,int(num),int(num))
		if int(num)<690:
			cnn.deepEstimate(basePath,64,int(num),int(num)+5)
		if int(num)>10:
			cnn.deepEstimate(basePath,64,int(num),int(num)-5)
	'''
main()
