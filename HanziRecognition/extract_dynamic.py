import cv2
import numpy as np
import os
from MSER_NMS import color_detect,rgb2hsv

def Dilate_Erode(img, size_dilate, size_erode):
    '''膨胀腐蚀处理'''
    # 指定核大小，如果效果不佳，可以试着将核调大
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, size_dilate)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, size_erode)

    # 对图像进行膨胀腐蚀处理
    img = cv2.erode(img, kernel_erode, anchor=(-1, -1), iterations=1)  # 腐蚀
    # img = cv2.dilate(img, kernel_dilate, anchor=(-1, -1), iterations=2)  # 膨胀
    # img = cv2.erode(img, kernel_erode, anchor=(-1, -1), iterations=1)  # 腐蚀
    img = cv2.dilate(img, kernel_dilate, anchor=(-1, -1), iterations=3)  # 膨胀
    # img = cv2.erode(img, kernel_erode, anchor=(-1, -1), iterations=2)  # 腐蚀

    return img

def extract_red(img, color_hsv):
    '''提取棋子区域'''
    src = None
    sourceImage = img.copy()
    qizi_Hanzi = None
    img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯模糊滤波器对图像进行模糊处理
    # cv2.imshow('sourceImage', sourceImage)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰色通道

    # 用于调整二值化参数
    # cv2.imshow('gray_img1', gray_img)
    # THRESHOLD_OF_GRAY = 45
    # _, gray_img1 = cv2.threshold(gray_img, THRESHOLD_OF_GRAY, 255, cv2.THRESH_BINARY)  # 对图像进行二值化操作
    # cv2.imshow('gray_img', gray_img1)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 转换为HSV空间

    # lower_red1 = np.array([0, 50, 0])  # 设定红色的阈值下限
    # upper_red1 = np.array([10, 255, 255])  # 设定红色的阈值上限
    # lower_red2 = np.array([165, 50, 0])  # 设定红色的阈值下限
    # upper_red2 = np.array([180, 255, 255])  # 设定红色的阈值上限
    lower_red1 = np.array([0, 50, 0])  # 设定红色的阈值下限
    upper_red1 = np.array([color_hsv[0]+125-180, 255, 255])  # 设定红色的阈值上限
    lower_red2 = np.array([color_hsv[0]+100, 50, 0])  # 设定红色的阈值下限
    upper_red2 = np.array([180, 255, 255])  # 设定红色的阈值上限

    # 消除噪声
    # plate_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)  # 设定掩膜取值范围
    plate_mask = cv2.inRange(hsv_img, lower_red1, upper_red1) + \
        cv2.inRange(hsv_img, lower_red2, upper_red2)  # 设定掩膜取值范围
    # hsv_mask = plate_mask.copy()
    # cv2.imshow('hsv_mask', hsv_mask)

    plate_mask = Dilate_Erode(plate_mask, size_dilate=(5, 5), size_erode=(5, 5))  # 膨胀腐蚀处理

    # 再对图像进行模糊处理
    plate_mask = cv2.medianBlur(plate_mask, 9)
    cv2.imshow('dilate_red', plate_mask)

    # 图像扶正
    # edge = cv2.Canny(plate_mask, 30, 120, 3)  # 边缘检测
    # cv2.imshow('edge',edge)

    contours, hier = cv2.findContours(
        plate_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 检测轮廓

    reg_plate = None
    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea,reverse=True)  # 根据轮廓面积从大到小排序
        for c in contours:

            peri = cv2.arcLength(c, True)  # 计算轮廓周长
            # print(peri)
            if peri < 400 or peri > 700:
                continue  # 周长不合要求，跳过

            # area = cv2.contourArea(c) # 计算轮廓面积
            # # print(area)
            # if area < 10000 or area > 40000:
            #     continue # 面积不合要求，跳过

            approx = cv2.approxPolyDP(c, 0.1*peri, True)  # 轮廓多边形拟合
            # print(approx)

            # # 调试用
            # for peak in approx:
            #     peak = peak[0] # 顶点坐标
            #     cv2.circle(sourceImage, tuple(peak), 10, (0, 0, 255),2) # 绘制顶点
            # cv2.imshow('ss',sourceImage)

            if len(approx) == 4:  # 轮廓为4个点表示找到棋子
                dist01square = (approx[0][0][0]-approx[1][0][0]
                                )**2 + (approx[0][0][1]-approx[1][0][1])**2
                dist03square = (approx[0][0][0]-approx[3][0][0]
                                )**2 + (approx[0][0][1]-approx[3][0][1])**2
                # print(dist01square,dist03square)

                if (float(dist01square)/dist03square > 1**2 and float(dist01square)/dist03square < 2.1**2) or (float(dist03square)/dist01square > 1**2 and float(dist03square)/dist01square < 2.1**2):
                    # 调试用
                    # for peak in approx:
                    #     peak = peak[0]  # 顶点坐标
                    #     cv2.circle(sourceImage, tuple(peak), 10, (0, 0, 255), 2)  # 绘制顶点
                    # cv2.imshow('ss', sourceImage)

                    src = np.float32(
                        [approx[0][0], approx[1][0], approx[2][0], approx[3][0]])  # 原图的四个顶点

                    width = 255
                    length = 455
                    side = 10

                    if dist01square < dist03square:
                        dst = np.float32([[0, 0], [0, width], [length, width], [length, 0]])  # 期望的四个顶点
                    else:
                        dst = np.float32([[length, 0], [0, 0], [0, width], [length, width]])  # 期望的四个顶点

                    m = cv2.getPerspectiveTransform(src, dst)  # 生成旋转矩阵
                    reg_plate = cv2.warpPerspective(gray_img, m, (length, width))  # 旋转后的图像

                    reg_plate = reg_plate[int(side):int(width-side), int(side):int(length-side)]  # 裁切掉边框干扰

                    reg_plate = cv2.normalize(reg_plate,None,0,255,cv2.NORM_MINMAX)             
                    reg_plate = cv2.equalizeHist(reg_plate)  # 直方图均衡化
                    
                    # cv2.imshow('reg_plate', reg_plate)

                    # print(peri)
                    break

    if reg_plate is None:
        print('未检测到棋子')
    else:
        print('检测到棋子')
        
        qizi_Hanzi = cv2.resize(reg_plate, (150,100), interpolation=cv2.INTER_AREA) # 对图像进行长宽比校正


        # _, qizi_Hanzi = cv2.threshold(qizi_Hanzi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # OTSU二值化

        # th, _ = cv2.threshold(qizi_Hanzi[:,0:75], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # OTSU二值化
        # _,qizi_Hanzi_left = cv2.threshold(qizi_Hanzi[:,0:75], int(th*0.85), 255, cv2.THRESH_BINARY)# OTSU二值化

        # th, _ = cv2.threshold(qizi_Hanzi[:,75:-1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # OTSU二值化
        # _,qizi_Hanzi_right = cv2.threshold(qizi_Hanzi[:,75:-1], int(th*0.85), 255, cv2.THRESH_BINARY)# OTSU二值化
        # qizi_Hanzi = np.hstack([qizi_Hanzi_left,qizi_Hanzi_right])


    return qizi_Hanzi,src


def ensure_dir(dir_path):
    '''生成文件夹'''
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError:
            pass


if __name__ == '__main__':
    imagePath = r"origin_img\red\zhadan\14.jpg"
    img = cv2.imread(imagePath)
    orig = img.copy()

    # hanzi_box = hanzi_box_detect(img)
    # print(hanzi_box)
    # for (x, y, x2, y2) in hanzi_box:
    #     cv2.rectangle(orig, (x, y), (x2, y2), (0, 255, 0), 2)
    # cv2.imshow("hanzi_box", orig)

    color = color_detect(img)
    # print("color:", color)
    h,s,v = rgb2hsv(color[2], color[1], color[0])
    print("hsv:", h,s,v)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
