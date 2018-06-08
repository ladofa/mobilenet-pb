import tensorflow as tf
import numpy as np
import cv2
from tensorflow.python.platform import gfile

def bin_predict(last_conv, training = False):
	pool = tf.layers.average_pooling2d(last_conv, 7, 1)
	dropout = tf.layers.dropout(pool, training=training)
	#conv2d = tf.layers.conv2d(dropout, 2, 1)
	sqeeze = tf.squeeze(dropout)
	return sqeeze

tf.GraphKeys.USEFUL = 'useful'

with tf.Session() as sess:

	#saver = tf.train.import_meta_graph("d:/models/mobilenet_v2/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt.meta")
	#saver.restore(sess, "d:/models/mobilenet_v2/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt")

	model_filename = 'd:/models/mobilenet_v2/mobilenet_v2_1.0_224/my.pb'
	with gfile.GFile(model_filename, 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	graph = sess.graph
	g_in = tf.import_graph_def(graph_def, name='test')
	logdir = 'd:/logs/mod'
	

	tensor_input = graph.get_tensor_by_name('test/input:0')
	tensor_output = graph.get_tensor_by_name('test/MobilenetV2/Predictions/Reshape_1:0')

	last_conv = graph.get_tensor_by_name("test/MobilenetV2/Conv_1/Relu6:0")

	#with tf.name_scope('my_logits'):
	#	logits = bin_predict(last_conv)

	#with tf.name_scope('my_predict'):
	#	labels = tf.placeholder(tf.int32, [None, 2])
	#	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
	#	optimizer = tf.train.AdamOptimizer()
	#	train_op = optimizer.minimize(loss=loss)

		

	train_writer = tf.summary.FileWriter('d:/logs/mod', graph)
	png = cv2.imread('c:/Users/ladofa/Desktop/Clipboard01.png')
	png = cv2.resize(png, (224, 224))
	cv2.imshow('asdf', png)
	cv2.waitKey(1)

	n_frame = np.asarray(png)
	in_image = n_frame / 255 - 0.5
	in_image = np.expand_dims(in_image, axis=0) 

	fake_in = np.ones([1, 224, 224, 3])
	output = sess.run(tensor_output, feed_dict={tensor_input:in_image})

	for i in range(output.shape[1]):
		print(str(i) + ' : ' + str(output[0, i]))

	print(output)