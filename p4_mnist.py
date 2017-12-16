#!/usr/bin/python
import sys, os

sys.path.append('/home/xcao')

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


# MNIST, FASHION MNIST and NOT MNIST dataset
#mnist = input_data.read_data_sets('./data/MNIST_data', one_hot=True)
mnist = input_data.read_data_sets('./data/fashion', one_hot=True)
#mnist = input_data.read_data_sets('./data/notMNIST', one_hot=True)

def main(file = None):
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
		y_ = tf.placeholder(tf.float32, [None, 10])
		
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

		# initialize variables
		sess.run(tf.initialize_all_variables())

		print("Start training.")

		# initial learning rate
		lr = 0.001

		# start training
		for epoch in range(1,51):

			# change the learing rate accordingly
			if epoch == 25:
				lr = 0.00001

			# iterations per each epoch
			for i in range(550):

				# obtain batch
				batch = mnist.train.next_batch(100)

				# learning rate decay
				lr = lr * 1/(1 + (lr /25.0) * epoch)

				if i%1000 == 0:

					# output the training accuracy
					train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], learning_rate: lr, keep_prob: 1.0})
					file.write("epoch %d, step %d, training accuracy %g\n"%(epoch, i, train_accuracy))

					# output the validatiaon accuracy
					t1 = correct_num.eval(feed_dict={x:mnist.validation.images[0:2500], y_: mnist.validation.labels[0:2500], learning_rate: lr, keep_prob: 1.0})
					t2 = correct_num.eval(feed_dict={x:mnist.validation.images[2500:5000], y_: mnist.validation.labels[2500:5000], learning_rate: lr, keep_prob: 1.0})
					file.write("epoch {0}, step {1}, validation accuracy {2}\n".format(epoch, i,(t1+t2)/10000))
				
				# run the training step
				t = sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], learning_rate: lr, keep_prob: 0.5})

		# output the final test result
		file.write("Final test accuracy %g\n"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

# run the training with file path
def run(outfilepath, iter):

	# open the outfile path
	with open(outfilepath,"w+") as f:

		# for each iteration run experimetns
		for i in range(iter):

			# runthe main training
			main(file = f)

			# reset the graph
			tf.reset_default_graph()

		# finish with indentation
		f.write("\n")

# if this file is caleld, run expeiment
if __name__ == '__main__':

	# set the iterations
	for i in range(1):
	
		# run experiments
		run("./out/FASHION_MNIST{0}.txt".format(i),1)


