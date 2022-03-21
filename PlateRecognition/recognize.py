#encoding:utf-8
import cv2
import numpy as np

#将图片转为灰度图像
img = cv2.imread('33.png')

#将原图做个备份
sourceImage = img.copy()

#高斯模糊滤波器对图像进行模糊处理
img = cv2.GaussianBlur(img, (3, 3), 0)

# cv2.imshow('sourceImage', sourceImage)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰色通道
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 转换为HSV空间

lower_blue = np.array([100, 100, 100])  # 设定蓝色的阈值下限
upper_blue = np.array([250, 255, 255])  # 设定蓝色的阈值上限

# 消除噪声
plate_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)  # 设定掩膜取值范围
blue_mask = plate_mask.copy()
# cv2.imshow('blue',plate_mask)

 
# 指定核大小，如果效果不佳，可以试着将核调大
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# 对图像进行膨胀腐蚀处理
# plate_mask = cv2.dilate(plate_mask, kernelY, anchor=(-1, -1), iterations=1)
plate_mask = cv2.dilate(plate_mask, kernel_dilate, anchor=(-1, -1), iterations=1)
# plate_mask = cv2.erode(plate_mask, kernel_erode, anchor=(-1, -1), iterations=2)


# 再对图像进行模糊处理
plate_mask = cv2.medianBlur(plate_mask, 9)
# cv2.imshow('dilate',plate_mask)

# 图像扶正
edge = cv2.Canny(plate_mask, 30, 120, 3) # 边缘检测
# cv2.imshow('edge',edge)


# 检测轮廓
# 输入的三个参数分别为：输入图像、层次类型、轮廓逼近方法
# 返回的两个返回值分别为：图轮廓、层次
contours, hier = cv2.findContours(plate_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

reg_plate = None
if len(contours)>0:
    contours = sorted(contours, key=cv2.contourArea, reverse=True) # 根据轮廓面积从大到小排序
    for c in contours:
        peri = cv2.arcLength(c, True) # 计算轮廓周长
        approx = cv2.approxPolyDP(c, 0.09*peri, True) # 轮廓多边形拟合

        # # 调试用
        # for peak in approx:
        #     peak = peak[0] # 顶点坐标
        #     print(peak)
        #     cv2.circle(sourceImage, tuple(peak), 10, (0, 0, 255),2) # 绘制顶点
        # cv2.imshow('ss',sourceImage)

        # 轮廓为4个点表示找到车牌
        if len(approx) == 4:
            src = np.float32([approx[0][0],approx[1][0],approx[2][0],approx[3][0]])
            width = 250
            length = 450
            side = 15
            dst = np.float32([[0, 0], [0, width],  [length, width],[length, 0]])
            m = cv2.getPerspectiveTransform(src, dst)
            reg_plate = cv2.warpPerspective(blue_mask, m, (length, width))
            # 裁切掉边框干扰
            reg_plate = reg_plate[int(side):int(width-side),int(side):int(length-side)]

if reg_plate is None:
    print('未检测到棋子')
else:
    print('检测到棋子')
    # cv2.imshow("reg_plate", reg_plate)

# 边缘检测
lpImage = cv2.Canny(reg_plate, 500, 200, 3)

# 对图像进行二值化操作
ret, thresh = cv2.threshold(lpImage.copy(), 127, 255, cv2.THRESH_BINARY)
cv2.imshow('thresh', thresh)

# 字符分割





# # 轮廓检测
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