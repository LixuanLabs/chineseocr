# coding=utf-8
import cv2


def show_image(_raw_image):
	"""
	显示图像
	:param _raw_image: 
	"""
	cv2.imshow("test", _raw_image)
	cv2.waitKey()


def to_threshold(_img_file):
	"""
	二值化图片
	:param _img_file: 
	:return: 
	"""
	img = cv2.imread(_img_file)

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
	return img


def cv_to_img_file(_raw_img, _route):
	"""
	保存opencv图片
	:param _raw_img: 
	:param _route: 
	"""
	cv2.imwrite(_route, _raw_img)
