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
import numpy as np
import time

class PID:
    '''实现PID控制器'''
    def __init__(self, sv, kp, ki, kd, max_out, min_out):
        self.sv = sv
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.min_out = min_out
        self.last_error = 0
        self.last_time = 0
        self.integral = 0
        self.derivative = 0
        self.output = 0

    def update(self, ratio, time):
        '''更新PID输出'''
        error = ratio - self.sv
        self.integral += error * (time - self.last_time)
        self.derivative = (error - self.last_error) / (time - self.last_time)
        self.output = self.kp * error + self.ki * self.integral + self.kd * self.derivative
        self.output = max(self.output, self.min_out)
        self.output = min(self.output, self.max_out)
        self.last_error = error
        self.last_time = time
        return self.output

def np_move_avg(a,n,mode="same"):
    return(np.convolve(a, np.ones((n,))/n, mode=mode))

def cal_ratio(img):
    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 计算灰度直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 255])
    hist = hist.reshape(-1)
    # hist[0] /= 8
    # hist[1] /= 2
    # hist = np_move_avg(hist, 3)
    # print(hist)

    plt.figure(0)
    plt.plot(hist)
    
    x = range(40)
    plt.fill_between(x, 0, hist[:40], facecolor=(12/255, 82/255, 160/255), alpha=0.5)
    x = range(40, 215)
    plt.fill_between(x, 0, hist[40:215], facecolor=(12/255, 82/255, 160/255), alpha=0.25)
    x = range(215,256)
    plt.fill_between(x, 0, hist[215:256], facecolor=(12/255, 82/255, 160/255), alpha=0.5)
    plt.vlines(39, 0, hist[39], colors='tab:blue', linestyles='dashed')
    plt.vlines(215, 0, hist[215], colors='tab:blue', linestyles='dashed')
    plt.show()
    
    # 计算极端情况的占比
    ratio = 1 - hist[40:215].sum()/hist.sum()
    return ratio

def need_light(ratio):
    '''
    判断光比，是否需要打开补光灯
    '''
    
    # 判断是否需要打开补光灯
    if ratio > 0.3:
        return True
    else:
        return False




if __name__ == '__main__':
    # 读入图像
    # img = cv2.imread(r'test_for_need_light.png')
    img = cv2.imread(r'origin_img\red\lvzhang\WIN_20220507_08_17_23_Pro.jpg')
    # img = cv2.imread(r'test3.png')
    # 显示图像
    cv2.imshow('img', img)
    ratio = cal_ratio(img)
    print(ratio)
    # 判断是否需要打开补光灯
    if need_light(ratio):
        print('需要打开补光灯')
    else:
        print('不需要打开补光灯')
    pid = PID(sv = 0.1, kp=0.5, ki=0.1, kd=0.1, max_out=1, min_out=0)
    print(pid.update(ratio, time.time()))
    cv2.waitKey(0)
    cv2.destroyAllWindows()