#encoding:utf-8
import cv2
import numpy as np

# def rotateClockWise90(img):
#     trans_img = cv2.transpose( img )
#     img90 = cv2.flip(trans_img, -1)
#     img90 = cv2.flip(img90, 1)
#     return img90



#将图片转为灰度图像
img = cv2.imread('22.png')
#将原图做个备份
sourceImage = img.copy()
cv2.imshow('sourceImage', sourceImage)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰色通道
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 转换为HSV空间

lower_blue = np.array([100, 100, 100])  # 设定蓝色的阈值下限
upper_blue = np.array([250, 255, 255])  # 设定蓝色的阈值上限
#  消除噪声
plate_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)  # 设定掩膜取值范围

cv2.imshow('blue',plate_mask)

#高斯模糊滤波器对图像进行模糊处理
img = cv2.GaussianBlur(img, (3, 3), 0)
# cv2.imshow('GaussianBlur',img)
 
#指定核大小，如果效果不佳，可以试着将核调大
kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
#对图像进行膨胀腐蚀处理
plate_mask = cv2.dilate(plate_mask, kernelY, anchor=(-1, -1), iterations=1)
plate_mask = cv2.dilate(plate_mask, kernelX, anchor=(-1, -1), iterations=1)

#再对图像进行模糊处理
plate_mask = cv2.medianBlur(plate_mask, 9)
cv2.imshow('dilate',plate_mask)
cv2.imshow('plate',plate_mask*gray_img)


# 图像扶正
# https://blog.csdn.net/HUXINY/article/details/89467344













 
#检测轮廓，
#输入的三个参数分别为：输入图像、层次类型、轮廓逼近方法
#因为这个函数会修改输入图像，所以上面的步骤使用copy函数将原图像做一份拷贝，再处理
#返回的两个返回值分别为：图轮廓、层次
# contours, hier = cv2.findContours(plate_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# for c in contours:
#     # 边界框
#     x, y, w, h = cv2.boundingRect(c)

#     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#     print('w' + str(w))
#     print('h' + str(h))
#     print(float(w)/h)
#     print('------')
#     #由于国内普通小车车牌的宽高比为3.14，所以，近似的认为，只要宽高比大于2.2且小于4的则认为是车牌
#     if (float(w)/h >= 2.2 and float(w)/h <= 4.0) or (float(h)/w >= 2.2 and float(h)/w <= 4.0):
#         #将车牌从原图中切割出来
#         lpImage = sourceImage[y:y+h, x:x+w]
#         if w<h:
#             lpImage = rotateClockWise90(lpImage)

#         cv2.imshow('lpImage',lpImage)
 
# if 'lpImage' not in dir():
#     print('未检测到车牌!')
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     exit()
 
# cv2.imshow('chepai', lpImage)
# #边缘检测
# lpImage = cv2.Canny(lpImage, 500, 200, 3)
# #对图像进行二值化操作
# ret, thresh = cv2.threshold(lpImage.copy(), 127, 255, cv2.THRESH_BINARY)
# #轮廓检测
# contours, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# i = 0
# lpchars = []
# for c in contours:
#     # 边界框
#     x, y, w, h = cv2.boundingRect(c)
#     cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 0, 0), 2)
 
#     print('w' + str(w))
#     print('h' + str(h))
#     print(float(w)/h)
#     print(str(0.8 * thresh.shape[0]))
#     print('------')
 
#     #根据比例和高判断轮廓是否字符
#     if float(w)/h >= 0.3 and float(w)/h <= 0.8 and h >= 0.6 * thresh.shape[0]:
#         #将车牌从原图中切割出来
#         lpImage2 = lpImage[y:y+h, x:x+w]
#         cv2.imshow(str(i), lpImage2)
#         i += 1
#         lpchars.append([x, y, w, h])
 
# cv2.imshow('sdd', thresh)
 
# if len(lpchars) < 1:
#     print('未检测到字符!')
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     exit()
 
# lpchars = np.array(lpchars)
# #对x坐标升序，这样，字符顺序就是对的了
# lpchars = lpchars[lpchars[:,0].argsort()]
# print(lpchars)
 
# #如果识别的字符小于7，说明汉字没识别出来，要单独识别汉字
# if len(lpchars) < 7:
#     aveWidth = 0
#     aveHeight = 0
#     aveY = 0
#     for index in lpchars:
#         aveY += index[1]
#         aveWidth += index[2]
#         aveHeight += index[3]
 
#     aveY = aveY/len(lpchars)
#     aveWidth = aveWidth/len(lpchars)
#     aveHeight = aveHeight/len(lpchars)
#     zhCharX = lpchars[0][0] - (lpchars[len(lpchars) - 1][0] - lpchars[0][0]) / (len(lpchars) - 1)
#     if zhCharX < 0:
#         zhCharX = 0
 
#     print(aveWidth)
#     print(aveHeight)
#     print(zhCharX)
#     print(aveY)
#     cv2.imshow('img', lpImage[int(aveY):int(aveY + aveHeight), int(zhCharX):int(zhCharX + aveWidth)])
 
 
 
cv2.waitKey(0)
cv2.destroyAllWindows()