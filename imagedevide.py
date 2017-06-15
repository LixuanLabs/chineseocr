# coding:utf8

import sys

import cv2
# import cv2.cv as cv
import numpy as np
from PIL import Image
import os
import time
import helper
import matplotlib.pyplot as plot
import correctimage


def preprocess(gray, filename='', image_root_path=''):
	# 1. Sobel算子，x方向求梯度
	sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=1)
	# 2. 二值化
	ret, binary = cv2.threshold(sobel, 0, 255,
								cv2.THRESH_OTSU + cv2.THRESH_BINARY)
	# binary = 255 - binary

	# 3. 膨胀和腐蚀操作的核函数
	element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
	element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
	element3 = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 1))

	# 4. 膨胀一次，让轮廓突出
	dilation = cv2.dilate(binary, element2, iterations=3)

	# 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
	# erosion = dilation
	erosion = cv2.erode(dilation, element1, iterations=2)

	# 6. 再次膨胀，让轮廓明显一些
	dilation2 = cv2.dilate(erosion, element3, iterations=2)

	return dilation2


def findTextRegion(img, src, filename='', image_root_path=''):
	region = list()

	# 1. 查找轮廓
	res = cv2.findContours(img, cv2.RETR_EXTERNAL,
						   cv2.CHAIN_APPROX_TC89_L1 )
	contours, hierarchy = res

	# print contours

	# 将框按照从左向右, 从上到下排序

	# pre_top = 0
	# 2. 筛选那些面积小的
	for i in range(len(contours)):
		cnt = contours[i]
		# 计算该轮廓的面积
		area = cv2.contourArea(cnt)

		# 面积小的都筛选掉
		if (area < 20):
			continue


		# 找到最小的矩形，该矩形可能有方向
		rect = cv2.minAreaRect(cnt)
		# print("rect is: ")
		# print( rect)

		# box是四个点的坐标
		box = cv2.cv.BoxPoints(rect)
		# box = [[rect[0][0],rect[0][1]], [rect[1][0], rect[1][1]]]
		box = np.int32(box)
		# print (box)
		h_ = box[:,1]
		w_ = box[:,0]
		# print (h_)
		# print (w_)
		w = abs(np.max(w_)-np.min(w_))
		h = abs(np.max(h_)-np.min(h_))
		# 排除太细的矩形
		if h > 100 :
			continue
		leftup = [np.min(h_), np.min(w_)]
		leftdown = [np.min(h_), np.max(w_)]
		rightup = [np.max(h_), np.min(w_)]
		rightdown = [np.max(h_), np.max(w_)]

		result = src[np.min(h_):np.max(h_), np.min(w_):np.max(w_)]
		# print (result)
		try:
			newimg = Image.fromarray(np.uint8(result))
		#     n = ''.join(filename.split('.')[:-1])
		#     #
		#     # # if not os.path.exists('img/res/img/temp/'+n):
		#     # #     os.mkdir('img/res/img/temp/'+n+'/res')
		#     thish = int(np.min(h_)/10/3)
		#     newimg.save('media/out/out/'+n+'_'+str(10000*thish+np.min(w_))+'.JPG')
		#     # newimg = cv2.imread(image_root_path+'media/out/out/'+n+'_'+str(10000*np.min(h_)+np.min(w_))+'.JPG')
		# #
		# #     box = np.int32(box)
		# #
		# #     region.append(box)
		#
		#     region.append('media/out/out/'+n+'_'+str(10000*thish+np.min(w_))+'.JPG')
			region.append(newimg)
			# region.append(result.astype(np.uint8))
		except Exception as e:
			print(e)

	print 'in findtextregion: ', len(region)

	return sorted(region)[:10]


def detect(img, filename='', image_root_path=''):
	# 1.  转化成灰度图
	# begin = time.time()
	try:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	except Exception as e:
		print (e)
		gray = img

	# 2. 形态学变换的预处理，得到可以查找矩形的图片
	dilation = preprocess(gray, filename, image_root_path)

	# 3. 查找和筛选文字区域
	# 返回一个包含需要图片的区域
	region = findTextRegion(dilation, img, filename, image_root_path)
	# 4. 用绿线画出这些找到的轮廓
	# for box in region:
	#     # print (box)
	#     cv2.drawContours(img, [box], 0, (127, 127, 127), 2)
	#     # cv2.rectangle(img, (box[2,0],box[2,1]), (box[0,0],box[0,1]), (255,0,0), 5)
	return region

def detectfromimgpath(img_file):
	return detect(cv2.imread(img_file),
				  filename=img_dir.split('/')[-1],
				  image_root_path=''.join(img_dir.split('/')[0:-1]))

def detectfromrawimage(_raw_image, img_file):
	"""
		从数据原图分割图片
	:param _raw_image: opencv读取的图片
	:param img_file: 文件的绝对路径
	:return: 返回分割后的图片
	"""
	return detect(_raw_image, filename=img_file.split('/')[-1], image_root_path=''.join(img_file.split('/')[0:-1]))

def processimagefiles(image_root_path):
	filenames = os.listdir(image_root_path)
	for filename in filenames:
		if filename.split(".")[-1].lower() in ("jpg", "png", "tiff"):
			regions = detect(cv2.imread(os.path.join(image_root_path, filename)),
							  filename=filename,
							  image_root_path=image_root_path)
			for region in regions:
				region.show()
				region.close()

def devide_image_by_horizon(_raw_image):
	# 去除红色像素并且二值化
	_raw_image = helper.cv_rid_red(_raw_image)
	img = helper.to_threshold_from_raw_image(_raw_image)
	img = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=1)
	img = correctimage.erode(img, x_axis=2)
	# img = correctimage.erode(img, x_axis=1)
	# helper.show_image(img)
	# img = cv2.convertScaleAbs(img)
	horizontal_sum = np.sum(img, axis=1)
	return extract_peek_ranges_from_array(horizontal_sum)

def extract_peek_ranges_from_array(array_vals, mininum_val = 511, mininum_range = 10, mininum_space=10):
	"""

	:param array_vals: 
	:param mininum_val: 
	:param mininum_range: 一个字符的大小
	:param mininum_space: 两个字符之间的距离
	:return: 
	"""
	start_i = None
	end_i = None
	end_i_list = []
	peek_ranges = []
	for i, val in enumerate(array_vals):
		if val > mininum_val and start_i is None:
			start_i = i
		elif val > mininum_val and start_i is not None:
			pass
		elif val < mininum_val and start_i is not None:
			end_i_list.append(i)
			if len(end_i_list) < mininum_space:
				end_i = i
			else:
				if end_i - start_i >= mininum_range:
					peek_ranges.append((start_i, end_i))
				start_i = None
				end_i_list = []
				end_i = None
		elif val < mininum_val and start_i is None:
			pass
		else:
			raise ValueError("cannot parse this case...")
	return peek_ranges

def rid_title_from_horizon(_raw_image, mininum_val = 511, mininum_size=8):
	"""

	:param _raw_image: 
	:param mininum_val: 
	:param mininum_size: 一个字符的宽度
	:return: 
	"""
	start_j = 0
	end_j = _raw_image.shape[1]
	start_j_list = []
	end_j_list = []
	img = helper.to_threshold_from_raw_image(_raw_image)
	# helper.show_image(img)
	# sobel_img = cv2.Sobel(img, cv2.CV_8U, 1, 1, ksize=1)
	# erode_img = correctimage.erode(img, x_axis=1)
	# erode_img = correctimage.erode(erode_img, x_axis=1)

	vertical_sum = np.sum(img, axis=0)
	# plot.plot(vertical_sum)
	# plot.show()
	for j, val in enumerate(vertical_sum):
		if val > mininum_val and start_j is 0:
			start_j = j
		elif val > mininum_val and start_j is not 0:
			end_j_list.append(j)
			if len(end_j_list) > mininum_size:
				end_j = j
		elif val < mininum_val and start_j is not 0:
			end_j_list = []
		elif val < mininum_val and start_j is 0:
			pass
		else:
			raise ValueError("can't parse this value")

	# end_j加10为微调
	end_j = _raw_image.shape[1] if (end_j + 10) > _raw_image.shape[1] else (end_j + 10)
	start_j = start_j if (start_j - 10) < 0 else (start_j - 10)
	return img[:, start_j: end_j]

def rid_word_from_a_line(_raw_image):
	vertical_sum = np.sum(_raw_image, axis=0)
	peek_range = extract_peek_ranges_from_array(vertical_sum, mininum_val=20, mininum_range=1, mininum_space=2)
	if len(peek_range):
		return 1, peek_range
	return 0, []
if __name__ == '__main__':
	# image_dir = "/Users/Alex/Desktop/test"
	# processimagefiles(image_dir)

	image_file = "/Users/Alex/Desktop/test/image00024.JPG"
	devide_image_by_horizon(cv2.imread(image_file))
