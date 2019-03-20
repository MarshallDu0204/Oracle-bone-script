import tensorflow as tf
import numpy as np
import cv2
from PIL import Image as Ige

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


def readData_single(path):
	path = path+"/train_set_cnn.tfrecords"

	filename_queue = tf.train.string_input_producer([path],num_epochs = 1,shuffle = True)

	reader = tf.TFRecordReader()

	_,serialized_example = reader.read(filename_queue)

	features = tf.parse_single_example(serialized_example,features = {
			'img1': tf.FixedLenFeature([], tf.string),
			'img2': tf.FixedLenFeature([], tf.string),
			'label': tf.FixedLenFeature([], tf.int32)
		})

	image1 = tf.decode_raw(features['img1'],tf.uint8)

	image1 = tf.reshape(image,[96,96,1])

	image2 = tf.decode_raw(features['img2'],tf.uint8)

	image2 = tf.reshape(image,[96,96,1])

	label = tf.cast(features['label'], tf.int32)

	return image1,image2,label


class CNN:
	def __init__(self):

		self.input_image1 = None

		self.input_image2 = None

		self.label = None

		self.keep_prob = tf.placeholder(dtype=tf.float32)

		self.lamb = tf.placeholder(dtype = tf.float32)

		self.prediction = None

		self.correct_prediction = None

		self.accurancy = None

		self.loss = None

		self.train_step = None

		
	def weight_variable(self,shape):
	    initial = tf.truncated_normal(shape,stddev=tf.sqrt(x = 2/(shape[0]*shape[1]*shape[2])))
	    tf.add_to_collection(name = 'loss',value=tf.contrib.layers.l2_regularizer(self.lamb)(initial))   
	    return tf.Variable(initial)

	def bias_variable(self,shape):
	    initial = tf.random_normal(shape=shape,dtype = tf.float32)
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

		self.label = tf.placeholder(dtype = tf.float32,shape = [batch_size,2])

		#2*96*96*1 --> 192*96*1 --> 96*48*32

		with tf.name_scope('first_convolution'):

			#-------merge img1 and img2 -----------

			mergeImg = self.merge_img(input_image1,input_image2)

			X = mergeImg

			#----- convolution -------

			w_conv = self.weight_variable([3,3,1,32])
			b_conv = self.bias_variable([32])

			X = tf.nn.relu(conv2d(X,w_conv)+b_conv)

			#----- maxpooling ---------

			X = self.max_pooling(X)


		#96*48*32 --> 48*24*64

		with tf.name_scope('second_convolution'):

			#------- convolution --------

			w_conv = self.weight_variable([3,3,32,64])
			b_conv = self.bias_variable([64])

			X = tf.nn.relu(conv2d(X,w_conv)+b_conv)

			#-------maxpooling----------

			X = self.max_pooling(X)

		#48*24*64 --> 24*12*128

		with tf.name_scope('third_convolution'):

			#-------- convolution -------

			w_conv = self.weight_variable([3,3,64,128])
			b_conv = self.bias_variable([128])

			X = tf.nn.relu(conv2d(X,w_conv)+b_conv)

			#-------maxpooling----------

			X = self.max_pooling(X)

			X = tf.nn.dropout(X,keep_prob = self.keep_prob)

		#hidden layer 1  24*12*128 --> 4096

		with tf.name_scope('hidden_layer_1'):

			#-------reshape---------

			X = tf.reshape(X,[-1,24*12*128])

			w_conv = self.weight_variable([batch_size*24*12*128,4096])
			b_conv = self.weight_variable([4096])

			X = tf.nn.relu(tf.matmul(X,w_conv)+b_conv)

		#hidden layer 2  4096 --> 4096

		with tf.name_scope('hidden_layer_2'):


			w_conv = self.weight_variable([4096,4096])
			b_conv = self.weight_variable([4096])

			X = tf.nn.relu(tf.matmul(X,w_conv)+b_conv)

			X = tf.nn.dropout(X,keep_prob = self.keep_prob)

		#final layer  4096 -- > 2

		with tf.name_scope('final_layer'):

			w_conv = self.weight_variable([4096,2])
			b_conv = self.bias_variable([2])

			X = tf.nn.softmax(tf.matmul(X,w_conv)+b_conv)



		#softmax loss

		with tf.name_scope('softmax'):

			self.loss = tf.reduce_mean(-tf.reduce_sum(input_label*tf.log(X),axis=1))


	    #accurancy

	    with tf.name_scope('accurancy'):

	    	self.correct_prediction = tf.equal(tf.argmax(X,1),tf.argmax(input_label,1)) 

			self.accurancy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) 



		#optimize

		with tf.name_scope('gradient_descent'):

			self.train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)



		def train(self,batch_size,path):

			ckpt_path = path+"/ckpt-cnn/model.ckpt"
		
			tf.summary.scalar("loss", self.loss)
			
			tf.summary.scalar('accuracy', self.accurancy)
			
			merged_summary = tf.summary.merge_all()

			model_dir = path+"/data/model"

			tb_dir = path+"/data/logs"

			all_parameters_saver = tf.train.Saver()


			with tf.Session() as sess:

				image1,image2,label = readData_single()

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


						lo,acc,summary = sess.run([self.loss,self.accurancy,merged_summary],feed_dict = {
								self.input_image1:example1,self.input_image2:example2,self.input_label:label,self.keep_prob:0.7,self.lamb:0.004
							})

						summary_writer.add_summary(summary, epoch)

						sess.run([self.train_step],feed_dict={
								self.input_image1: example1,self.input_image2:example2,self.input_label: abel,self.keep_prob: 1.0,
								self.lamb: 0.004
							})

						epoch+=1

						if epoch%10 == 0:
							print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))


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
			img1 = convertToBinary(img1)

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
			img2 = convertToBinary(img2)

			data2 = img2

			newImg2 = []

			i=0
			while i!=batch_size:
				newImg2.append(data2)
				i+=1
			
			data2 = np.reshape(a=newImg2, newshape=(batch_size,96,96,1))



			ckpt_path = path+"/ckpt-cnn/model.ckpt"

			all_parameters_saver = tf.train.Saver()

			with tf.Session() as sess:  
				sess.run(tf.global_variables_initializer())
				sess.run(tf.local_variables_initializer())
				all_parameters_saver.restore(sess=sess, save_path=ckpt_path)
				predict_result = sess.run(
								tf.argmax(input=self.prediction, axis=3), 
								feed_dict={
									self.input_image1: data1,self.input_image2:data2
									self.keep_prob: 1.0, self.lamb: 0.004
								}
							)

				predict_result = predict_result[0]
				
				print(predict_result) 
			print('Done prediction')


def main():
	basePath = "C:/Users/24400/Desktop"
	cnn = CNN()
	cnn.setup_network(16)
	#cnn.train(16,basePath)
	cnn.estimate(16,basePath)



		