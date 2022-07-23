'''
判断光比，是否需要打开补光灯

树莓派从摄像头读入当前图像后，将图像转为灰度图并计算图像的灰度直方
图，当灰度图中处于灰度级215~255（高光部分）的像素数目与处于灰度级0~40
（阴影部分）的像素数目的比值（光比）大于一预设阈值时，认为当前环境为大
光比环境。此时，树莓派给继电器输出高电平，继电器闭合，LED 灯带亮，给棋
子文字面补光。否则，继电器断开，LED 灯带灭。
'''
import cv2
import matplotlib.pyplot as plt

def need_light(img):
    '''
    判断光比，是否需要打开补光灯
    '''
    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 计算灰度直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 255])
    # print(hist)
    # plt.figure(0)
    # plt.plot(hist)
    # plt.show()
    # 计算光比
    light_ratio = hist[215:-1].sum() / hist[0:40].sum()
    # print(light_ratio)
    # 计算极端情况的占比
    ratio = 1 - hist[40:215].sum()/hist.sum()
    # print(ratio)
    
    # 判断是否需要打开补光灯
    if light_ratio > 10 and ratio > 0.3:
        return True
    else:
        return False

if __name__ == '__main__':
    # 读入图像
    img = cv2.imread(r'origin_img\green\dilei\1.jpg')
    # 显示图像
    cv2.imshow('img', img)
    # 判断是否需要打开补光灯
    if need_light(img):
        print('需要打开补光灯')
    else:
        print('不需要打开补光灯')
    cv2.waitKey(0)
    cv2.destroyAllWindows()