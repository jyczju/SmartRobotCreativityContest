from tkinter import N
import cv2
import numpy as np

def getHProjection(image):
    '''水平投影'''    
    #图像高与宽
    (h,w)=image.shape
    #长度与图像高度一致的数组
    H = [0]*h
    #循环统计每一行白色像素的个数
    for y in range(h):
        for x in range(w):
            if image[y,x] == 0:
                H[y]+=1
    
    # # 绘制水平投影图像，调试用
    # hProjection = np.zeros(image.shape,np.uint8)
    # for y in range(h):
    #     for x in range(H[y]):
    #         hProjection[y,x] = 255
    # cv2.imshow('hProjection2',hProjection)
 
    return H


def getVProjection(image):
    '''垂直投影'''
    #图像高与宽
    (h,w) = image.shape
    #长度与图像宽度一致的数组
    W = [0]*w
    #循环统计每一列白色像素的个数
    for x in range(w):
        for y in range(h):
            if image[y,x] == 0:
                W[x]+=1
    
    # # 绘制垂直平投影图像，调试用
    # vProjection = np.zeros(image.shape,np.uint8);
    # for x in range(w):
    #     for y in range(h-W[x],h):
    #         vProjection[y,x] = 255
    # cv2.imshow('vProjection',vProjection)

    return W


img = cv2.imread('44.png') # 读取图片
sourceImage = img.copy() # 将原图做个备份

img = cv2.GaussianBlur(img, (3, 3), 0) # 高斯模糊滤波器对图像进行模糊处理
cv2.imshow('sourceImage', sourceImage)

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
plate_mask = cv2.dilate(plate_mask, kernel_dilate, anchor=(-1, -1), iterations=1) # 膨胀
# plate_mask = cv2.erode(plate_mask, kernel_erode, anchor=(-1, -1), iterations=2) # 腐蚀

# 再对图像进行模糊处理
plate_mask = cv2.medianBlur(plate_mask, 9)
# cv2.imshow('dilate',plate_mask)

# 图像扶正
edge = cv2.Canny(plate_mask, 30, 120, 3) # 边缘检测
# cv2.imshow('edge',edge)


contours, hier = cv2.findContours(plate_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 检测轮廓

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

        # 轮廓为4个点表示找到棋子
        if len(approx) == 4: # 此处还可加上长宽比判据
            src = np.float32([approx[0][0],approx[1][0],approx[2][0],approx[3][0]]) # 原图的四个顶点
            width = 250
            length = 450
            side = 15
            dst = np.float32([[0,0], [0,width], [length,width], [length,0]]) # 期望的四个顶点
            m = cv2.getPerspectiveTransform(src, dst) # 生成旋转矩阵
            reg_plate = cv2.warpPerspective(blue_mask, m, (length, width)) # 旋转后的图像
            _, reg_plate = cv2.threshold(reg_plate, 127, 255, cv2.THRESH_BINARY) # 对图像进行二值化操作
            reg_plate = reg_plate[int(side):int(width-side),int(side):int(length-side)] # 裁切掉边框干扰

if reg_plate is None:
    print('未检测到棋子')
else:
    print('检测到棋子')
    # cv2.imshow("reg_plate", reg_plate)

    lpImage = cv2.Canny(reg_plate, 500, 200, 3) # 边缘检测
    ret, thresh = cv2.threshold(lpImage.copy(), 127, 255, cv2.THRESH_BINARY) # 对图像进行二值化操作
    # cv2.imshow('thresh', thresh)
    
    # 字符分割
    H = getHProjection(reg_plate) # 水平投影
    H_Start = None
    H_End = None
    for i in range(len(H)):
        if H[i] > 0 and H_Start is None:
            H_Start = i # 寻找开始点
        if H[i] <= 0 and H_Start is not None:
            H_End =i # 寻找结束点
        if H_Start is not None and H_End is not None:
            if (H_End-H_Start)<0.5*len(H): # 排除干扰
                continue
            else:
                break
    
    # 缩减上下间距
    reg_plate_H = reg_plate[H_Start:H_End,:]
    First_Hanzi_H = thresh[H_Start:H_End,:]
    
    W = getVProjection(reg_plate_H) # 垂直投影
    W_Start = None
    W_End = None
    for i in range(len(W)):
        if W[i] > 0 and W_Start is None:
            W_Start = i # 寻找开始点
        if W[i] <= 0 and W_Start is not None:
            W_End =i # 寻找结束点
        if W_Start is not None and W_End is not None:
            if (W_End-W_Start)<0.35*len(W): # 排除偏旁干扰
                continue
            else:
                break

    # 根据确定的位置分割出第一个字符
    First_Hanzi = First_Hanzi_H[:,W_Start:W_End]
    cv2.imshow('First_Hanzi',First_Hanzi)

cv2.waitKey(0)
cv2.destroyAllWindows()