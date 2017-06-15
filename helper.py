# coding=utf-8
import cv2
import numpy as np

def show_image(_raw_image):
	"""
	显示图像
	:param _raw_image: 
	"""
	cv2.namedWindow("test", cv2.WINDOW_NORMAL)
	cv2.imshow("test", _raw_image)
	cv2.waitKey()
	cv2.destroyAllWindows()


def to_threshold(_img_file):
	"""
	二值化图片
	:param _img_file: 
	:return: 
	"""
	img = cv2.imread(_img_file)
	return to_threshold_from_raw_image(img)

def to_threshold_from_raw_image(_raw_image):
	gray = cv2.cvtColor(_raw_image, cv2.COLOR_BGR2GRAY)
	img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
	return img

def from_image_to_cv(_raw_img):
	"""
	将Image读取的图片转换为opencv读取的图片
	:param _raw_img: 
	:return: 
	"""
	img = cv2.cvtColor(np.array(_raw_img), cv2.COLOR_RGB2BGR)
	return img

def cv_to_img_file(_raw_img, _route):
	"""
	保存opencv图片
	:param _raw_img: 
	:param _route: 
	"""
	cv2.imwrite(_route, _raw_img)

def cv_rotate_img(_filename, _rotate_angle):
	"""
	旋转图片
	:param _filename: 文件的路径
	:param _rotate_angle: 旋转的角度
	:return: 旋转后的图片
	"""
	img = cv2.imread(_filename)
	height, width = img.shape[:2]
	mat = cv2.getRotationMatrix2D((width / 2, height / 2), _rotate_angle, 1)
	rotated_img = cv2.warpAffine(img, mat, (height, width))
	return rotated_img

def cv_rid_red(_raw_image):
	for i in range(_raw_image.shape[0]):
		for j in range(_raw_image.shape[1]):
			if _raw_image[i, j, 2] > 255 * 0.8:
				_raw_image[i][j] = [255, 255, 255]
			else:
				pass
	return _raw_image