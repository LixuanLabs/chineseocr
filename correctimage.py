# coding:utf-8
import cv2
import numpy as np
import math
from PIL import Image


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
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,1))
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
	contours, hierarchy = cv2.findContours(raw_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
			line_length.append(length)
			angle = math.atan(h_w / v_h)
			if not angle == 0:
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

if __name__ == "__main__":
	filename = "/Users/Alex/Desktop/image00009.JPG"
	img = threshold(filename)
	img = dilate(img)
	img = erode(img)
	img = dilate(img)
	img = dilate(img)
	img = dilate(img)
	img = cv2.medianBlur(img, 11)
	rotate_angle= caculate_rotate_angle(img)
	print("rotate_angle:", rotate_angle)
	img = Image.open(filename)
	img = img.rotate(rotate_angle)
	img.show()
