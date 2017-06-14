# coding:utf-8
import cv2
import numpy as np
import math
from PIL import Image
import helper
import os


def threshold(_filename):
	"""
	二值化图片
	:param _filename: 
	:return: 
	"""
	_img = cv2.imread(_filename, cv2.IMREAD_GRAYSCALE)
	adaptive_threshold = cv2.adaptiveThreshold(
		_img,
		255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		cv2.THRESH_BINARY_INV, 21, 5
	)
	return adaptive_threshold

def dilate(raw_img):
	"""
		膨胀图片
	:param raw_img: 
	:return: 
	"""
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,1))
	raw_img = cv2.dilate(raw_img, kernel)
	return raw_img

def erode(raw_img):
	"""
		腐蚀图片
	:param raw_img: 
	:return: 
	"""
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,1))
	raw_img = cv2.erode(raw_img, kernel)
	return raw_img

def caculate_rotate_angle(raw_image):
	"""
	图片纠正
	:param raw_image:
	:return: 旋转的度数，正值为逆时针旋转，负值为正时针旋转
	"""
	contours, hierarchy = cv2.findContours(raw_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
	angle_weight = []
	line_length = []
	max_y_set_x = []
	min_y_set_x = []
	for index in range(len(contours)):
		cnt = contours[index]
		# 计算该轮廓的面积
		area = cv2.contourArea(cnt)
		# 面积小的都筛选掉
		if area < 20:
			continue
		# 找到最小的矩形，该矩形可能有方向
		rect = cv2.minAreaRect(cnt)
		# box是四个点的坐标，box的第一个坐标表示的是最高点的坐标，其余点是逆时针方向
		box = cv2.cv.BoxPoints(rect)
		box = np.int0(box)
		# 将Y坐标最大和最小的横坐标存储
		max_y_set_x.append(box[0][0])
		min_y_set_x.append(box[2][0])
		if box[0][0] > box[2][0]:
			# 右上偏高
			h_w = abs(box[2][0] - box[3][0])
			v_h = abs(box[2][1] - box[3][1])
		else:
			# 左上偏高
			h_w = abs(box[2][0] - box[1][0])
			v_h = abs(box[2][1] - box[1][1])

		if not h_w == 0 and not v_h == 0:
			length = math.sqrt(math.pow(h_w, 2) + math.pow(v_h, 2))
			if length > 100:
				# 弧度值
				arc_angle = math.atan2(v_h, h_w)
				# 转换为角度
				angle = math.degrees(arc_angle)
				if not angle == 0:
					line_length.append(length)
					angle_weight.append(angle * length)
	# 判断偏转的方向
	if np.sum(max_y_set_x) > np.sum(min_y_set_x):
		prefix = 1
	else:
		prefix = -1
	try:
		angle = prefix * (np.sum(angle_weight) / np.sum(line_length))
	except ArithmeticError:
		angle = 0
	finally:
		return angle

def precorrect(_filename):
	"""
	纠正的整套流程
	:param _filename: 
	:return: 返回纠正后的图片
	"""
	_img = threshold(_filename)
	# _img = erode(_img)
	_img = dilate(_img)

	_img = dilate(_img)
	_img = dilate(_img)
	_img = dilate(_img)
	_img = dilate(_img)
	# _img = cv2.medianBlur(_img, 11)
	_rotate_angle = caculate_rotate_angle(_img)
	return _filename, _rotate_angle

def predir(_img_dir):
	filenames = os.listdir(_img_dir)
	for filename in filenames:
		print(filename)
		if filename.split(".")[-1].lower() in ("jpg", "png", "tiff"):
			file, rotate_angle = precorrect(os.path.join(_img_dir, filename))
			print ("filename, torate_angle", filename, rotate_angle)
			img = Image.open(file)
			img.rotate(rotate_angle).show()
			img.close()


if __name__ == "__main__":
	filename = "/Users/Alex/Desktop/test/image00020.JPG"
	file, rotate_angle = precorrect(filename)
	print("rotate_angle", rotate_angle)
	img = Image.open(file)
	img = img.rotate(rotate_angle)
	img.show()
	# img_dir = "/Users/Alex/Desktop/test"
	# predir(img_dir)

