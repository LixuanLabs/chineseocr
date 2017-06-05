# coding:utf8

import sys

import cv2
# import cv2.cv as cv
import numpy as np
from PIL import Image
import os
import time

# image_root_path = 'img'
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SAVE_DIR = os.path.join(ROOT_DIR, "temp_data_img")


def showImage(img):
	cv2.namedWindow('test_')
	cv2.imshow('test_', img)
	cv2.waitKey(1000)
	cv2.destroyWindow('test_')


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

	# showImage(binary)
	# showImage(dilation)
	# showImage(erosion)
	# showImage(dilation2)

	# # 7. 存储中间图片
	# filenamepre = ''.join(filename.split('.')[:-1])
	# if not os.path.exists(image_root_path+'/res/img/temp/' + filenamepre):
	#     os.mkdir(image_root_path+'/res/img/temp/' + filenamepre )
	#     os.mkdir(image_root_path+'/res/img/temp/'+ filenamepre+ '/temp')
	#     os.mkdir((image_root_path+'/res/img/temp/'+filenamepre+'/res'))
	# cv2.imwrite(image_root_path+'/res/img/temp/'+filenamepre+"/temp/1.png", binary)
	# cv2.imwrite(image_root_path+'/res/img/temp/'+filenamepre+"/temp/2.png", dilation)
	# cv2.imwrite(image_root_path+'/res/img/temp/'+filenamepre+"/temp/3.png", erosion)
	# cv2.imwrite(image_root_path+'/res/img/temp/'+filenamepre+"/temp/4.png", dilation2)

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

		# # 轮廓近似，作用很小
		# epsilon = 0.001 * cv2.arcLength(cnt, True)
		# approx = cv2.approxPolyDP(cnt, epsilon, True)

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
			# n = ''.join(filename.split('.')[:-1])
		#     #
		#     # # if not os.path.exists('img/res/img/temp/'+n):
		#     # #     os.mkdir('img/res/img/temp/'+n+'/res')
			#
			# thish = int(np.min(h_)/10/3)
			# newimg.save(os.path.join(SAVE_DIR,n+'_'+str(10000*thish+np.min(w_))+'.JPG'))
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
	# print time.time() - begin
	# 4. 用绿线画出这些找到的轮廓
	# for box in region:
	#     # print (box)
	#     cv2.drawContours(img, [box], 0, (127, 127, 127), 2)
	#     # cv2.rectangle(img, (box[2,0],box[2,1]), (box[0,0],box[0,1]), (255,0,0), 5)
	#
	# showImage(img)
	# plt.imshow(img)
	# plt.show()
	#
	# # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
	# # cv2.imshow("img", img)
	#
	# # 带轮廓的图片
	# cv2.imwrite(image_root_path+'/res/img/'+filename, img)
	#
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	return region

def detectfromimgpath(img_dir):
	return detect(cv2.imread(img_dir),
				  filename=img_dir.split('/')[-1],
				  image_root_path=''.join(img_dir.split('/')[0:-1]))
def processimagefiles(image_root_path):
	try:
		os.mkdir(image_root_path+'/res/img')
		os.mkdir(image_root_path+'/res/img/temp')
	except Exception as e:
		print(e)
	for f in os.listdir(image_root_path):
		print (f)
		f_ext = f.split('.')[-1]
		if f_ext not in ( 'JPG', 'jpg', 'PNG', 'png', 'TIF', 'tif'):
			continue
		img = cv2.imread(image_root_path+'/'+f)

		detect(img, f, image_root_path)


if __name__ == '__main__':
	filename = 'testimg/3.JPG'
	img = cv2.imread(filename)

	print detect(img, )

	# filename = 'image00043.JPG'
	# im = cv2.imread('img/'+filename)
	#
	# detect(im, filename, 'img')
	# filename = 'a0000001A.tif'
	# im = cv2.imread('img2/'+filename)
	# detect(im, filename)

	# 读取文件
	# imagePath = 'img/image00005.JPG'
	# files = os.listdir(image_root_path+'/')
	# count=0
	# for f in files:
	#     if f.split('.')[-1] not in ('JPG', 'jpg', 'PNG', 'png', 'tif'):
	#         continue
	#     img = cv2.imread(image_root_path+'/'+f)
	#     # print(img)
	#     detect(img, f, 'img')
	#     count += 1
		# if count>0:
		#     break
	# processimagefiles('img')
	# processimagefiles('img2')
	# processimagefiles('img3')
	# processimagefiles('img4')
