# coding=utf-8
import numpy as np
import inference
import tensorflow as tf
import cv2
import re
import os
import sys
sys.path.append("..")
import dataset
import correctimage
import helper
import imagedevide
import codecs

IMAGE_HEIGHT = 45
IMAGE_WIDTH = 45
IMAGE_CHANNEL = 1

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_model_dir = os.path.join(root_dir, "models")
_label_file = os.path.join(root_dir, "3500.txt")


def production(_filename):
	# 偏正图片
	_filename, rotate_angle = correctimage.precorrect(_filename)
	rotated_img = helper.cv_rotate_img(_filename, rotate_angle)
	medium = rotated_img.shape[0] // 2
	# 获取水平前五行图片的上线和下线
	peek_array = imagedevide.devide_image_by_horizon(rotated_img)[:5]
	# 读取label文件
	label_content = dataset.read_label(_label_file).decode("utf-8")
	# re.compile(ur'[^\u4e00-\u9fa5]')
	reuse = False
	for region in peek_array:
		# 切割标题
		raw_img = imagedevide.rid_title_from_horizon(rotated_img[region[0]:region[1], medium-250:medium+250])
		# 从标题中抽出每一个字
		flag, word_imgs = imagedevide.rid_word_from_a_line(raw_img)
		if flag:
			for index, word_img in enumerate(word_imgs):
				with tf.Session() as sess:
					ckpt = tf.train.get_checkpoint_state(_model_dir)
					if ckpt and ckpt.model_checkpoint_path:
						# 通过文件名获取模型保存时的迭代
						global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
						img = raw_img[:,word_img[0]: word_img[1]]

						img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
						img = img.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL)
						# helper.show_image(img)
						img = np.array([img], dtype=np.float32)
						y = inference.inference(img, reuse=reuse)
						reuse = True
						saver = tf.train.Saver()
						saver.restore(sess, ckpt.model_checkpoint_path)
						result = sess.run(y)[0]
						index_max = tf.argmax(result, 0).eval()
						if result[index_max] > 4:
							print("After %s steps, result is %s" % (global_step, label_content[index_max]))
						else:
							print ("no match")

if __name__ == "__main__":
	filename = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../applyimages/test1.png")
	print (filename)
	production(filename)
