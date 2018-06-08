import tensorflow as tf
import mobilenet_v2


#image_ph = tf.placeholder(images.dtype, [None, 224, 224])
#labels_ph = tf.placeholder(labels.dtype, [None, 1])


def parser(record):
	keys_to_features = {
		"train/image": tf.FixedLenFeature((224, 224), tf.float32, default_value=tf.zeros([], dtype=tf.float32)),
		"train/label": tf.FixedLenFeature((), tf.int64, default_value=1)
	}
	parsed = tf.parse_single_example(record, keys_to_features)
	
	image = tf.reshape(image, [299, 299, 1])
	label = tf.cast(parsed["label"], tf.int32)

	return {"image_data": parsed["image_data"]}, label

with tf.Session() as sess:

	
	# For simplicity we just decode jpeg inside tensorflow.
	# But one can provide any input obviously.
	dataset = tf.data.TFRecordDataset('01.tfrecords')
	dataset = dataset.repeat()
	dataset = dataset.shuffle(32)
	
	batch_size = 32
	dataset = dataset.batch(batch_size)
	iterator = dataset.make_initializable_iterator()
	next_element = iterator.get_next()

	feature = {'train/image': tf.FixedLenFeature([batch_size, 224, 224, 3], tf.float32),
               'train/label': tf.FixedLenFeature([batch_size, 1], tf.int64)}

	sess.run(iterator.initializer)
	raw = sess.run(next_element)
	data = tf.parse_example(raw, feature)

	# Note: arg_scope is optional for inference.
	with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=True)):
		logits, endpoints = mobilenet_v2.mobilenet(data['train/image'])
	# Restore using exponential moving average since it produces (1.5-2%) higher 
	# accuracy
	ema = tf.train.ExponentialMovingAverage(0.999)
	vars = ema.variables_to_restore()

	saver = tf.train.Saver(vars)
	checkpoint = 'd:/models/mobilenet_v2/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt'
	saver.restore(sess,  checkpoint)

	#loss.. and optimizer
	loss = tf.losses.sparse_softmax_cross_entropy(data['train/label'], logits)
	op = tf.train.AdamOptimizer()
	op.minimize(loss)

	sess.run(op)

	print(op)

	

	

	





	