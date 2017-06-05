# coding: utf-8
import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def yStart(grey):
    m,n = grey.size
    for j in xrange(n):
        for i in xrange(m):
            if grey.getpixel((i,j)) == 0:
                return j

def yEnd(grey):
    m,n = grey.size
    for j in xrange(n-1,-1,-1):
        for i in xrange(m):
            if grey.getpixel((i,j)) == 0:
                return j

def xStart(grey):
    m,n = grey.size
    for i in xrange(m):
        for j in xrange(n):
            if grey.getpixel((i,j)) == 0:
                return i

def xEnd(grey):
    m,n = grey.size
    for i in xrange(m-1,-1,-1):
        for j in xrange(n):
            if grey.getpixel((i,j)) == 0:
                return i

def xBlank(grey):
    m,n = grey.size
    blanks = []
    for i in xrange(m):
        for j in xrange(n):
            if grey.getpixel((i,j)) == 0:
                break
        if j == n-1:
            blanks.append(i)
    return blanks

def yBlank(grey):
    m,n = grey.size
    blanks = []
    for j in xrange(n):
        for i in xrange(m):
            if grey.getpixel((i,j)) == 0:
                break
        if i == m-1:
            blanks.append(j)
    return blanks

def getWordsList():
    f = open('3500.txt')
    line = f.read().decode('utf-8')
    # wordslist = line.split(' ')
    f.close()
    print len(line)
    return line

count = 0
wordslist = []
def getWordsByBlank(img,path):
    '''根据行列的空白取图片，效果不错'''
    global count
    global wordslist
    grey = img.split()[0]
    xblank = xBlank(grey)
    yblank = yBlank(grey)   
    #连续的空白像素可能不止一个，但我们只保留连续区域的第一个空白像素和最后一个空白像素，作为文字的起点和终点
    xblank = [xblank[i] for i in xrange(len(xblank)) if i == 0 or i == len(xblank)-1 or not (xblank[i]==xblank[i-1]+1 and xblank[i]==xblank[i+1]-1)]
    yblank = [yblank[i] for i in xrange(len(yblank)) if i == 0 or i == len(yblank)-1 or not (yblank[i]==yblank[i-1]+1 and yblank[i]==yblank[i+1]-1)]    
    for j in xrange(len(yblank)/2):
        for i in xrange(len(xblank)/2):
            area = (xblank[i*2],yblank[j*2],xblank[i*2+1]+2,yblank[j*2]+45)#这里固定字的大小是32个像素
            #area = (xblank[i*2],yblank[j*2],xblank[i*2+1],yblank[j*2+1])
            word = img.crop(area)
            word.save(path+wordslist[count]+'.png')
            count += 1
            if count >= len(wordslist):
                return

def getWordsFormImg(imgName,path):
    png = Image.open(imgName,'r')
    img = png.convert('1')
    grey = img.split()[0]
    #先剪出文字区域
    area = (xStart(grey)-1,yStart(grey)-1,xEnd(grey)+2,yEnd(grey)+2)
    img = img.crop(area)  
    getWordsByBlank(img,path)

def getWrods():
    global wordslist
    wordslist = getWordsList()
    base_dir = "/Users/Alex/Desktop"
    imgs = ["35001.png", "35002.png","35003.png"]
    for img in imgs:        
        getWordsFormImg(os.path.join(base_dir, img),'words/')

if __name__ == "__main__":
	getWrods()


# base_dir = "/Users/Alex/Desktop"
# path_image = os.path.join(base_dir, "3500个常用汉字.png")
# image_color = cv2.imread(path_image)
# new_shape = (image_color.shape[1] // 3, image_color.shape[0] // 3)
# image_color = cv2.resize(image_color, new_shape)
# image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
# adaptive_threshold = cv2.adaptiveThreshold(
# 	image,
# 	255,
# 	cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
# 	cv2.THRESH_BINARY_INV, 11, 2
# )
# cv2.imshow("image", adaptive_threshold)
# cv2.waitKey()
# 计算水平值的大小
# horizontal_sum = np.sum(adaptive_threshold, axis=1)
# plt.plot(horizontal_sum, range(horizontal_sum.shape[0]))
# plt.gca().invert_yaxis()
# plt.show()
