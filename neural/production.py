# coding=utf-8
import numpy as np
import inference
import tensorflow as tf
import os
import sys
sys.path.append("..")
import dataset

IMAGE_HEIGHT = 45
IMAGE_WIDTH = 45
IMAGE_CHANNEL = 1

root_dir = os.path.join(os.path.abspath(__file__), "..")
_model_dir = os.path.join(root_dir, "models")
_label_file = os.path.join(root_dir, "3500.txt")


def production(_raw_image):
	img = _raw_image.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL)
	img = np.array([img], dtype=np.float32)
	label_content = dataset.read_label(_label_file)
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(_model_dir)
		if ckpt and ckpt.model_checkpoint_path:
			# 通过文件名获取模型保存时的迭代
			global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
			y = inference.inference(img, reuse=False)
			saver = tf.train.Saver()
			saver.restore(sess, ckpt.model_checkpoint_path)
			result = sess.run(y)[0]
			index_max = tf.argmax(result, 0).eval()
			if result[index_max] > 4:
				print("After %s steps, result is %s" % (global_step, label_content[index_max].decode("utf-8")))
