#!/usr/bin/python
import sys, os

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# MNIST, FASHION MNIST and NOT MNIST dataset
mnist = input_data.read_data_sets('./data/MNIST_data', one_hot=True)
fmnist = input_data.read_data_sets('./data/fashion', one_hot=True)
nmnist = input_data.read_data_sets('./data/notMNIST', one_hot=True)

"""
A function that takes length, data and lables and outputs randomized batch
"""
def next_batch(num, data, labels):
	'''
	Return a total of `num` random samples and labels. 
	'''
	# arange the index
	idx = np.arange(0 , len(data))

	# shuffle the index
	np.random.shuffle(idx)

	# cut the index to the size of the batch
	idx = idx[:num]

	# select data according to the index
	data_shuffle = [data[ i] for i in idx]
	labels_shuffle = [labels[ i] for i in idx]

	# return the data and label
	return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def expandLabel(lst, dataset_index, num_datasets):
	"""
	num_datasets 3 -> 3 datasets
	dataset_index 1 -> first dataset
	"""
	l = []

	for i in range(len(lst)):
		elem = [0.0]*(num_datasets*10)
		elem[((dataset_index-1)*10):(dataset_index*10)] = lst[i]
		l.append(elem)

	return l

# Training, Validation and Testing Images
_train_img 	= mnist.train.images.tolist() + nmnist.train.images.tolist() #nmnist.train.images.tolist() +
_val_img	= mnist.validation.images.tolist() + nmnist.validation.images.tolist() #nmnist.validation.images.tolist() +
_test_img	= mnist.test.images.tolist() + nmnist.test.images.tolist() #nmnist.test.images.tolist() +

# Training labels expanded to 30 classes
mnist_train_label 	= expandLabel(mnist.train.labels.tolist(),1,3)
fmnist_train_label = expandLabel(fmnist.train.labels.tolist(),2,3)
nmnist_train_label = expandLabel(nmnist.train.labels.tolist(),3,3)

# Validataion labels expanded to 30 classes
mnist_val_label  = expandLabel(mnist.validation.labels.tolist(),1,3)
fmnist_val_label = expandLabel(fmnist.validation.labels.tolist(),2,3)
nmnist_val_label = expandLabel(nmnist.validation.labels.tolist(),3,3)

# Testing labels expanded to 30 classes
mnist_test_label = expandLabel(mnist.test.labels,1,3)
fmnist_test_label = expandLabel(fmnist.test.labels,2,3)
nmnist_test_label = expandLabel(nmnist.test.labels,3,3)

# Concatinationg labels
_train_label 	=  mnist_train_label + fmnist_train_label + nmnist_train_label
_val_label		=  mnist_val_label + fmnist_val_label + nmnist_val_label
_test_label		=  mnist_test_label + fmnist_test_label + nmnist_test_label


# main function that is called for the training
def main(filepath = None):
	with tf.Session() as sess:

		# weight initialzier
		def weight_variable(shape):
			initial = tf.truncated_normal(shape, stddev = 0.1)
			return tf.Variable(initial)
		
		# bias initializer
		def bias_variable(shape):
			initial = tf.constant(0.1, shape = shape)
			return tf.Variable(initial)
		
		# Computes a 2-D convolution with stride of 1
		def conv2d(x, W):
			return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')
		
		def max_pool_2x2(x):
			return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
		
		# input
		x = tf.placeholder(tf.float32, shape = [None, 784])

		# learning_rate
		learning_rate = tf.placeholder(tf.float32, shape=[])

		# learning_rate
		keep_prob = tf.placeholder(tf.float32, shape=[])

		# reshape the image with some * 28 * 28
		x_image = tf.reshape(x, [-1,28,28,1])
		
		# the correct answer y_
		y_ = tf.placeholder(tf.float32, [None, 20])
		

		### 1st layer ###
		W_conv0 = weight_variable([3,3,1,96])
		b_conv0 = bias_variable([96])
		h_conv0 = tf.contrib.layers.batch_norm(tf.nn.conv2d(x_image, W_conv0, strides = [1,1,1,1], padding = 'SAME') + b_conv0)
		h_conv0 = tf.nn.relu(h_conv0)

		### 2nd layer ###
		W_conv1 = weight_variable([3,3,96,96])
		b_conv1 = bias_variable([96])
		h_conv1 = tf.contrib.layers.batch_norm(tf.nn.conv2d(h_conv0, W_conv1, strides = [1,1,1,1], padding = 'SAME') + b_conv1)
		h_conv1 = tf.nn.relu(h_conv1)

		### 3rd layer ###
		W_conv2 = weight_variable([3,3,96,96])
		b_conv2 = bias_variable([96])
		h_conv2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(h_conv1, W_conv2, strides = [1,2,2,1], padding = 'SAME') + b_conv2)
		h_conv2 = tf.nn.relu(h_conv2)
		h_conv2 = tf.nn.dropout(h_conv2,keep_prob)

		### 4th layer ###
		W_conv3 = weight_variable([3,3,96,192])
		b_conv3 = bias_variable([192])
		h_conv3 = tf.contrib.layers.batch_norm(tf.nn.conv2d(h_conv2, W_conv3, strides = [1,1,1,1], padding = 'SAME') + b_conv3)
		h_conv3 = tf.nn.relu(h_conv3)

		### 5th layer ###
		W_conv4 = weight_variable([3,3,192,192])
		b_conv4 = bias_variable([192])
		h_conv4 = tf.contrib.layers.batch_norm(tf.nn.conv2d(h_conv3, W_conv4, strides = [1,1,1,1], padding = 'SAME') + b_conv4)
		h_conv4 = tf.nn.relu(h_conv4)

		### 6th layer ###
		W_conv5 = weight_variable([3,3,192,192])
		b_conv5 = bias_variable([192])
		h_conv5 = tf.contrib.layers.batch_norm(tf.nn.conv2d(h_conv4, W_conv5, strides = [1,2,2,1], padding = 'SAME') + b_conv5)
		h_conv5 = tf.nn.relu(h_conv5)
		h_conv5 = tf.nn.dropout(h_conv5,keep_prob)

		### 7th layer ###
		W_conv6 = weight_variable([3,3,192,192])
		b_conv6 = bias_variable([192])
		h_conv6 = tf.contrib.layers.batch_norm(tf.nn.conv2d(h_conv5, W_conv6, strides = [1,1,1,1], padding = 'VALID') + b_conv6)
		h_conv6 = tf.nn.relu(h_conv6)	

		### 8th layer ###
		W_conv7= weight_variable([1,1,192,192])
		b_conv7 = bias_variable([192])
		h_conv7 = tf.contrib.layers.batch_norm(tf.nn.conv2d(h_conv6, W_conv7, strides = [1,1,1,1], padding = 'SAME') + b_conv7)
		h_conv7 = tf.nn.relu(h_conv7)

		### 9th layer ###
		W_conv8= weight_variable([1,1,192,20])
		b_conv8 = bias_variable([20])
		h_conv8 = tf.contrib.layers.batch_norm(tf.nn.conv2d(h_conv7, W_conv8, strides = [1,1,1,1], padding = 'SAME') + b_conv8)
		h_conv8 = tf.nn.relu(h_conv8)

		# gap h_pool2
		out = tf.reduce_mean(h_conv8, reduction_indices=[1, 2], name="avg_pool")
		
		# cross entropy comparing y_ and y_conv
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=out))
		
		# train step with adam optimizer
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
		
		# check if they are same
		correct_prediction = tf.equal(tf.argmax(out,1), tf.argmax(y_,1))
		
		# accuracy
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		# correct num
		correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

		sess.run(tf.initialize_all_variables())

		print("Start training.")

		# initial learning rate
		lr = 0.001

		# the number of epoches
		ep = 100

		# size of a batch
		bs = 100

		# the number of iterations per each epoch
		max_steps = 550

		# the training image after 50 epoches
		train_image_after_50 = mnist.train.images.tolist()

		# Start training
		for epoch in range(ep):

			# change the learning rate when the epoch is in 24, 74, 0 or 49.
			if epoch in [24,74]:
				lr = 0.00001
			elif epoch in [0,49]:
				lr = 0.001

			# training steps
			for i in range(max_steps):

				# obtain the batch
				batch = None
				if (epoch < 50):
					batch = next_batch(bs, _train_img, _train_label)
				else:
					batch = next_batch(bs, train_image_after_50, mnist_train_label)

				# learning rate decay
				lr = lr * 1/(1 + (lr /25.0) * epoch)

				# print out validation accuracy and training accuracy to a file
				if(i == 0):

					# training and validation
					train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], learning_rate: lr, keep_prob: 1.0})
					t1 = correct_num.eval(feed_dict={x:_val_img[0:2500], 	y_: _val_label[0:2500],  keep_prob: 1.0})
					t2 = correct_num.eval(feed_dict={x:_val_img[2500:5000], 	y_: _val_label[2500:5000], keep_prob: 1.0})

					# write them to the file
					with open(filepath+"EP{0}.txt".format(epoch),"w+") as file:
						file.write("epoch %d, step %d, training accuracy %g\n"%(epoch, i, train_accuracy))
						file.write("epoch {0}, step {1}, test accuracy {2}\n".format(epoch, i,(t1+t2)/5000))
		
				# run the training step
				t = sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], learning_rate: lr, keep_prob: 0.5})

		# testing image and testing label
		tstimg = mnist.test.images.tolist()
		tstlbl = mnist_test_label

		# output the test accuracy
		t1 = correct_num.eval(feed_dict={x:tstimg[0:2500], 		y_: tstlbl[0:2500], 		keep_prob: 1.0})
		t2 = correct_num.eval(feed_dict={x:tstimg[2500:5000], 	y_: tstlbl[2500:5000], 		keep_prob: 1.0})
		t3 = correct_num.eval(feed_dict={x:tstimg[5000:7500], 	y_: tstlbl[5000:7500], 		keep_prob: 1.0})
		t4 = correct_num.eval(feed_dict={x:tstimg[7500:10000],	y_: tstlbl[7500:10000], 	keep_prob: 1.0})
		with open(filepath+"FINAL.txt".format(),"w+") as file:
			file.write("Final test accuracy {0}\n".format((t1+t2+t3+t4)/10000))

# run with file path
def run(outfilepath):
	main(filepath = outfilepath)

# run task
if __name__ == '__main__':
	run("./out/MFN_M/MFN_M_")

