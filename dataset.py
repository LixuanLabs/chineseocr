# coding: utf-8
from __future__ import print_function
import os
import tensorflow as tf
from PIL import Image
import cv2
from six.moves import cPickle as Pickle

import helper

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(ROOT_DIR, "words")
DATASET_DIR = os.path.join(ROOT_DIR, 'datasets')
DATASET_GROUP = 1000
RESIZE_HEIGHT = 45
RESIZE_WIDTH = 45
IMAGE_DEEP = 1


def generate():
	"""
		生成数据集
	:return: 
	"""
	filenames = os.listdir(IMAGE_DIR)
	# 过滤图片文件
	for file in filenames:
		if not os.path.splitext(file)[1].lower() in (".png", ".jpg", ".bmp"):
			filenames.remove(file)
	# 生成数据集
	for order in range((len(filenames) // DATASET_GROUP) + 1):
		train_image = []
		train_label = []
		for index, filename in enumerate(filenames[order*DATASET_GROUP:order*DATASET_GROUP+DATASET_GROUP]):
			train_image, train_label = deal_image(filename, train_image, train_label, order * DATASET_GROUP + index)
		# 保存文件
		save = {
			"train_image": train_image,
			"train_label": train_label
		}
		f = open(os.path.join(DATASET_DIR, "train_image_"+ str(order) +".pickle"), "wb")
		Pickle.dump(save, f, Pickle.HIGHEST_PROTOCOL)
		f.close()

def read(filename):
	"""
	:parameter 数据集文件
	:return: 图片，标题
	"""
	f = open(os.path.join(DATASET_DIR, filename), "rb")
	dataset_file = Pickle.load(f)
	train_image = dataset_file["train_image"]
	train_label = dataset_file["train_label"]
	# train_image = tf.cast(train_image, tf.float32)
	# train_label = tf.cast(train_label, tf.int32)
	return train_image, train_label

def read_label(_label_file):
	"""
	label文件中的内容
	:param _label_file: label文件的路径
	:return: 
	"""
	f = open(_label_file, "r")
	lines = f.readlines()
	f.close()
	return lines

def deal_image(filename, train_image, train_label, file_index):
	"""
	:param file_index: 
	:param filename: 
	:param train_image: 
	:param train_label: 
	:return: 
	"""
	img = helper.to_threshold(os.path.join(IMAGE_DIR, filename))
	img = cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation=cv2.INTER_CUBIC)
	img = img.reshape(RESIZE_HEIGHT, RESIZE_WIDTH, IMAGE_DEEP)
	train_image.append(img)
	train_label.append(file_index)
	return train_image, train_label


if __name__ == "__main__":
	generate()
