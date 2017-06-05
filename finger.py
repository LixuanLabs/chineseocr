# coding: utf-8
from __future__ import print_function
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os






def identity(filename):
	"""
	:param filename: 
	:return: 
	"""
	is_figure = False
	img = cv2.imread(filename)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if not (abs(int(img[i, j, 2]) - int(img[i, j, 1])) > 20 and img[i, j, 2] > 235 and (
				int(img[i, j, 0]) - int(img[i, j, 1])) > 0):
				# if (img[i, j, 2] - img[i, j, 1]) < 70:
				img[i, j, :] = [255, 255, 255]
			# elif img[i, j, 1] < 210:
			# img[i, j, :] = [255, 255, 255]
	cv2.imshow("image", img)
	cv2.waitKey()
	# 灰度化
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# 二值化
	img = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img[i][j] != 255:
				# print(img[i][j])
				is_figure = True

	return is_figure


def test_identity(image_dir):
	"""
		测试
	"""
	filenames = os.listdir(image_dir)
	try:
		for i in filenames:
			print(i, ": ", identity(os.path.join(testDir, i)))
	except Exception as e:
		print("not open")
		raise


def main(argv=None):
	filename = "/Users/Alex/Desktop/文字识别文件/陈刚妨害公务/1/image00001.JPG"
	identity(filename)


if __name__ == "__main__":
	test_dir = "/Users/Alex/Desktop/文字识别文件/陈刚妨害公务/1/"
	test_identity(test_dir)
	# print(identity(filename))
