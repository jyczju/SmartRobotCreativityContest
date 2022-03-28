import cv2
import numpy as np
import os


def getHProjection(img):
    '''水平投影'''
    # 图像高与宽
    (h, w) = img.shape
    # 长度与图像高度一致的数组
    H = [0]*h
    # 循环统计每一行黑色像素的个数
    for y in range(h):
        for x in range(w):
            if img[y, x] <= 70:
                H[y] += 1

    # # 绘制水平投影图像，调试用
    # hProjection = np.zeros(image.shape,np.uint8)
    # for y in range(h):
    #     for x in range(H[y]):
    #         hProjection[y,x] = 255
    # cv2.imshow('hProjection2',hProjection)

    return H


def getVProjection(img):
    '''垂直投影'''
    # 图像高与宽
    (h, w) = img.shape
    # 长度与图像宽度一致的数组
    W = [0]*w
    # 循环统计每一列黑色像素的个数
    for x in range(w):
        for y in range(h):
            if img[y, x] <= 70:
                W[x] += 1

    # # 绘制垂直平投影图像，调试用
    # vProjection = np.zeros(image.shape,np.uint8);
    # for x in range(w):
    #     for y in range(h-W[x],h):
    #         vProjection[y,x] = 255
    # cv2.imshow('vProjection',vProjection)

    return W


def Cut_H(img, H):
    '''缩减上下间距'''
    H_Start = None
    H_End = None
    for i in range(len(H)):
        if H[i] > 0 and H_Start is None:
            H_Start = i  # 寻找开始点
        if H[i] <= 0 and H_Start is not None:
            H_End = i  # 寻找结束点
        if H_Start is not None and H_End is not None:
            if (H_End-H_Start) < 0.8*len(H):  # 排除干扰
                continue
            else:
                break

    if H_Start is not None and H_End is not None:
        if (H_End-H_Start) < 0.8*len(H):  # 排除偏旁干扰
            H_End = len(H)-1
        # 缩减上下间距
        img = img[H_Start:H_End, :]
        # First_Hanzi_H = thresh[H_Start:H_End, :]
    return img


def Cut_W(img, W):
    '''缩减左右间距'''
    W_Start = None
    W_End = None
    for i in range(len(W)):
        if W[i] > 0 and W_Start is None:
            W_Start = i  # 寻找开始点
        if W[i] <= 0 and W_Start is not None:
            W_End = i  # 寻找结束点
        if W_Start is not None and W_End is not None:
            # if (W_End-W_Start)<0.35*len(W): # 排除偏旁干扰
            if (W_End-W_Start) < 0.8*len(W):  # 排除偏旁干扰
                continue
            else:
                break

    if W_Start is not None and W_End is not None:
        if (W_End-W_Start) < 0.8*len(W):  # 排除偏旁干扰
            W_End = len(W)-1
        # 根据确定的位置分割出第一个字符
        # First_Hanzi = First_Hanzi_H[:,W_Start:W_End] # 空心字
        img = img[:, W_Start:W_End]  # 实心字
    return img


def Revise_HW(img, ah, aw):
    '''对图像进行长宽比校正'''
    h = img.shape[0]
    w = img.shape[1]
    src = np.float32([[0, 0], [0, h], [w, h], [w, 0]])  # 原图的四个顶点
    dst = np.float32([[0, 0], [0, ah], [aw, ah], [aw, 0]])  # 期望的四个顶点
    m = cv2.getPerspectiveTransform(src, dst)  # 生成旋转矩阵
    img = cv2.warpPerspective(img, m, (aw, ah))  # 旋转后的图像
    return img


def Dilate_Erode(img, size_dilate, size_erode):
    '''膨胀腐蚀处理'''
    # 指定核大小，如果效果不佳，可以试着将核调大
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, size_dilate)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, size_erode)

    # 对图像进行膨胀腐蚀处理
    img = cv2.erode(img, kernel_erode, anchor=(-1, -1), iterations=1)  # 腐蚀
    img = cv2.dilate(img, kernel_dilate, anchor=(-1, -1), iterations=2)  # 膨胀
    img = cv2.erode(img, kernel_erode, anchor=(-1, -1), iterations=1)  # 腐蚀
    # plate_mask = cv2.Canny(img, 30, 120, 3)
    img = cv2.dilate(img, kernel_dilate, anchor=(-1, -1), iterations=2)  # 膨胀
    img = cv2.erode(img, kernel_erode, anchor=(-1, -1), iterations=2)  # 腐蚀
    # plate_mask = cv2.dilate(img, kernel_dilate,anchor=(-1, -1), iterations=2)  # 膨胀
    # plate_mask = cv2.erode(img, kernel_erode,anchor=(-1, -1), iterations=4)  # 腐蚀
    # plate_mask = cv2.dilate(img, kernel_dilate,anchor=(-1, -1), iterations=5)  # 膨胀
    # plate_mask = cv2.erode(img, kernel_erode,anchor=(-1, -1), iterations=4)  # 腐蚀
    # plate_mask = cv2.dilate(img, kernel_dilate,anchor=(-1, -1), iterations=5)  # 膨胀
    # plate_mask = cv2.erode(img, kernel_erode, anchor=(-1, -1), iterations=2) # 腐蚀
    return img


def extract_red(img):
    '''提取棋子区域'''
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

    # lower_blue = np.array([100, 30, 70])  # 设定蓝色的阈值下限
    # upper_blue = np.array([250, 235, 255])  # 设定蓝色的阈值上限
    lower_red1 = np.array([0, 50, 50])  # 设定红色的阈值下限
    upper_red1 = np.array([5, 250, 255])  # 设定红色的阈值上限
    lower_red2 = np.array([175, 50, 50])  # 设定红色的阈值下限
    upper_red2 = np.array([180, 250, 255])  # 设定红色的阈值上限

    # 消除噪声
    # plate_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)  # 设定掩膜取值范围
    plate_mask = cv2.inRange(hsv_img, lower_red1, upper_red1) + \
        cv2.inRange(hsv_img, lower_red2, upper_red2)  # 设定掩膜取值范围
    # hsv_mask = plate_mask.copy()
    # cv2.imshow('hsv_mask', hsv_mask)

    plate_mask = Dilate_Erode(plate_mask, size_dilate=(
        5, 5), size_erode=(5, 5))  # 膨胀腐蚀处理

    # 再对图像进行模糊处理
    plate_mask = cv2.medianBlur(plate_mask, 9)
    cv2.imshow('dilate', plate_mask)

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
                    for peak in approx:
                        peak = peak[0]  # 顶点坐标
                        cv2.circle(sourceImage, tuple(peak), 10, (0, 0, 255), 2)  # 绘制顶点
                    cv2.imshow('ss', sourceImage)

                    src = np.float32(
                        [approx[0][0], approx[1][0], approx[2][0], approx[3][0]])  # 原图的四个顶点

                    width = 255
                    length = 455
                    side = 5

                    if dist01square < dist03square:
                        dst = np.float32([[0, 0], [0, width], [length, width], [length, 0]])  # 期望的四个顶点
                    else:
                        dst = np.float32([[length, 0], [0, 0], [0, width], [length, width]])  # 期望的四个顶点

                    m = cv2.getPerspectiveTransform(src, dst)  # 生成旋转矩阵
                    reg_plate = cv2.warpPerspective(
                        gray_img, m, (length, width))  # 旋转后的图像

                    # _, reg_plate = cv2.threshold(reg_plate, THRESHOLD_OF_GRAY, 255, cv2.THRESH_BINARY)  # 对图像进行二值化操作
                    reg_plate = cv2.equalizeHist(reg_plate)  # 直方图均衡化
                    # reg_plate = cv2.medianBlur(reg_plate,5)

                    reg_plate = reg_plate[int(side):int(width-side), int(side):int(length-side)]  # 裁切掉边框干扰

                    # cv2.imshow('reg_plate', reg_plate)

                    # print(peri)
                    break

    if reg_plate is None:
        print('未检测到棋子')
    else:
        print('检测到棋子')

        # 字符分割
        H = getHProjection(reg_plate)  # 水平投影
        reg_plate_H = Cut_H(reg_plate, H)  # 缩减上下间距
        # cv2.imshow('reg_plate_H',reg_plate_H)

        W = getVProjection(reg_plate_H)  # 垂直投影
        qizi_Hanzi = Cut_W(reg_plate_H, W)  # 缩减左右间距

        qizi_Hanzi = Revise_HW(qizi_Hanzi, ah=100, aw=200)  # 对图像进行长宽比校正

    return qizi_Hanzi


def extract_green(img):
    '''提取棋子区域'''
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

    lower_green = np.array([35, 43, 46])  # 设定绿色的阈值下限
    upper_green = np.array([77, 255, 255])  # 设定绿色的阈值上限

    # 消除噪声
    plate_mask = cv2.inRange(hsv_img, lower_green, upper_green)  # 设定掩膜取值范围

    # hsv_mask = plate_mask.copy()
    # cv2.imshow('hsv_mask', hsv_mask)

    plate_mask = Dilate_Erode(plate_mask, size_dilate=(
        5, 5), size_erode=(5, 5))  # 膨胀腐蚀处理

    # 再对图像进行模糊处理
    plate_mask = cv2.medianBlur(plate_mask, 9)
    cv2.imshow('dilate', plate_mask)

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
                    for peak in approx:
                        peak = peak[0]  # 顶点坐标
                        cv2.circle(sourceImage, tuple(peak),
                                   10, (0, 0, 255), 2)  # 绘制顶点
                    cv2.imshow('ss', sourceImage)

                    src = np.float32(
                        [approx[0][0], approx[1][0], approx[2][0], approx[3][0]])  # 原图的四个顶点

                    width = 255
                    length = 455
                    side = 5

                    if dist01square < dist03square:
                        dst = np.float32(
                            [[0, 0], [0, width], [length, width], [length, 0]])  # 期望的四个顶点
                    else:
                        dst = np.float32([[length, 0], [0, 0], [0, width], [
                                         length, width]])  # 期望的四个顶点

                    m = cv2.getPerspectiveTransform(src, dst)  # 生成旋转矩阵
                    reg_plate = cv2.warpPerspective(
                        gray_img, m, (length, width))  # 旋转后的图像

                    # _, reg_plate = cv2.threshold(reg_plate, THRESHOLD_OF_GRAY, 255, cv2.THRESH_BINARY)  # 对图像进行二值化操作
                    reg_plate = cv2.equalizeHist(reg_plate)  # 直方图均衡化
                    # reg_plate = cv2.medianBlur(reg_plate,5)

                    reg_plate = reg_plate[int(side):int(
                        width-side), int(side):int(length-side)]  # 裁切掉边框干扰

                    # cv2.imshow('reg_plate', reg_plate)

                    # print(peri)
                    break

    if reg_plate is None:
        print('未检测到棋子')
    else:
        print('检测到棋子')

        # 字符分割
        H = getHProjection(reg_plate)  # 水平投影
        reg_plate_H = Cut_H(reg_plate, H)  # 缩减上下间距
        # cv2.imshow('reg_plate_H',reg_plate_H)

        W = getVProjection(reg_plate_H)  # 垂直投影
        qizi_Hanzi = Cut_W(reg_plate_H, W)  # 缩减左右间距

        qizi_Hanzi = Revise_HW(qizi_Hanzi, ah=100, aw=200)  # 对图像进行长宽比校正

    return qizi_Hanzi


def ensure_dir(dir_path):
    '''生成文件夹'''
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError:
            pass


if __name__ == '__main__':

    qizi = ['dilei', 'gongbin', 'junqi', 'junzhang', 'lianzhang', 'lvzhang',
            'paizhang', 'shizhang', 'siling', 'tuanzhang', 'yinzhang', 'zhadan']


    # 提取红色棋子
    print('extract red qizi')
    for i in range(0, 12):
        print(qizi[i])
        save_dir = './extract_img/' + qizi[i] # 保存文件夹
        img_dir = './origin_img/red/' + qizi[i] # 来源文件夹

        for _, _, files in os.walk(img_dir):
            # 遍历文件
            # print(files)
            for f in files:
                img_file_dir = img_dir + '/' + f
                ensure_dir(save_dir)
                save_file_dir = save_dir + '/ex_red_' + f
                img = cv2.imread(img_file_dir)  # 读取图片
                # cv2.imshow('img', img)
                red_Hanzi = extract_red(img)
                if red_Hanzi is None:
                    print('Failed')
                else:
                    print('Success')
                    cv2.imwrite(save_file_dir, red_Hanzi)

    
    # # 提取绿色棋子
    # print('extract green qizi')
    # for i in range(0, 12):
    #     print(qizi[i])
    #     save_dir = './extract_img/' + qizi[i] # 保存文件夹
    #     img_dir = './origin_img/green/' + qizi[i] # 来源文件夹

    #     for _, _, files in os.walk(img_dir):
    #         # 遍历文件
    #         # print(files)
    #         for f in files:
    #             img_file_dir = img_dir + '/' + f
    #             ensure_dir(save_dir)
    #             save_file_dir = save_dir + '/ex_green_' + f
    #             img = cv2.imread(img_file_dir)  # 读取图片
    #             # cv2.imshow('img', img)
    #             green_Hanzi = extract_green(img)
    #             if green_Hanzi is None:
    #                 print('Failed')
    #             else:
    #                 print('Success')
    #                 # cv2.imshow('First_Hanzi', First_Hanzi)
    #                 cv2.imwrite(save_file_dir, green_Hanzi)



            
    # img = cv2.imread('./origin_img/red/gongbin/4.jpg')  # 读取图片
    # name_of_img = './extract_img/gongbin/ex_red_4.jpg'
    # sourceImage = img.copy()  # 将原图做个备份

    # First_Hanzi = extract_red(img)
    # if First_Hanzi is None:
    #     print('提取棋子失败')
    # else:
    #     print('提取棋子成功')
    #     cv2.imshow('First_Hanzi', First_Hanzi)
    #     cv2.imwrite(name_of_img, First_Hanzi)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
