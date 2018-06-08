from random import shuffle
import glob
import os
import cv2
import numpy as np
import tensorflow as tf
shuffle_data = True


def image2arr(image):
	image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)

	cv2.imshow('asdf', image)
	cv2.waitKey(0)

	n_frame = np.asarray(image)
	in_image = n_frame / 255 - 0.5
	#in_image = np.expand_dims(in_image, axis=0)
	return in_image


cat0_names = ['0', '2', '4', '6', '8']
cat1_names = ['1', '3', '5', '7' , '9']
allowed_exts = ['.jpg', '.JPG', '.png', '.PNG']
base_dir = 'd:\\dataset\\mnistasjpg\\images'
writer = tf.python_io.TFRecordWriter('asdf.tfrecords')
w = [x[0] for x in os.walk(base_dir)]
for sub_dir in w[1:3]:
	print(sub_dir)
	if any([key in sub_dir for key in cat0_names]): #cat0에 속한 디렉토리라면
		label = 0
	else:
		label = 1

	for file in os.listdir(sub_dir): #디렉토리 내의 모든 파일에 대해서
		print(sub_dir + '\\' + file)
		#이미지 파일이 아니면 건너뛴다.
		_, file_extension = os.path.splitext(file)
		if not any([allowed_ext == file_extension for allowed_ext in allowed_exts]):
			print('bad file found : ' + file)
			continue

		#이미지를 불러온다.
		image = cv2.imread(sub_dir + '\\' + file)
		image_array = image2arr(image)
		
		

		#피쳐로 변환
		label_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
		image_feature = tf.train.Feature(float_list=tf.train.FloatList(value=image_array.reshape(-1)))
		feature = {'train/label' : label_feature,
			 'train/image' : image_feature}
		example = tf.train.Example(features = tf.train.Features(feature=feature))
		writer.write(example.SerializeToString())
writer.close()
sys.stdout.flush()


