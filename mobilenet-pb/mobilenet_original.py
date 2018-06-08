import tensorflow as tf
import mobilenet_v2

with tf.Session() as sess:


	# For simplicity we just decode jpeg inside tensorflow.
	# But one can provide any input obviously.
	file_input = tf.placeholder(tf.string, ())

	image = tf.image.decode_jpeg(tf.read_file(file_input))

	images = tf.expand_dims(image, 0)
	images = tf.cast(images, tf.float32) / 128.  - 1
	images.set_shape((None, None, None, 3))
	images = tf.image.resize_images(images, (224, 224))

	# Note: arg_scope is optional for inference.
	with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=True)):
		logits, endpoints = mobilenet_v2.mobilenet(images)
	# Restore using exponential moving average since it produces (1.5-2%) higher 
	# accuracy
	ema = tf.train.ExponentialMovingAverage(0.999)
	vars = ema.variables_to_restore()

	saver = tf.train.Saver(vars)
	checkpoint = 'd:/models/mobilenet_v2/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt'
	saver.restore(sess,  checkpoint)
	x = endpoints['Predictions'].eval(feed_dict={file_input: 'c:/Users/ladofa/Desktop/Clipboard01.png'})

	train_writer = tf.summary.FileWriter('d:/logs/mod', sess.graph)

	print(x)
	