import cv2
import extract
import compare
import tflite_predict
import platform
import time
import serial
from UI import show_UI
if platform.system() == 'Windows':
    import tensorflow as tf
elif platform.system() == 'Linux':
    import tflite_runtime.interpreter as tflite         ####界定使用的系统

compareDict = {
    'green_win': 'A',
    'red_win': 'B',
    'equal_siling': 'C',
    'red_siling_killed_by_zhadan': 'D',
    'green_siling_killed_by_zhadan': 'E',
    'equal': 'F',
    'red_siling_killed_by_dilei': 'G',
    'green_siling_killed_by_dilei': 'H',
    'red_kill_green': 'I',
    'green_kill_red': 'J',
    'red_kill_green': 'K',
    'green_kill_red': 'L'
}                         ####对比较结果进行一个编号！再加四个M,N,O,P，分别是红绿方连续击杀和棋盘上少六枚棋子

piecenumber = {
    'gongbin'   : 1, 
    'paizhang'  : 2,
    'lianzhang' : 3,
    'yinzhang'  : 4,
    'tuanzhang' : 5,
    'lvzhang'   : 7,
    'shizhang'  : 8,
    'junzhang'  : 11,
    'siling'    : 13,               ####大子的话，把司令和军长算入就好了吧！
    'dilei'     : 3,
    'zhadan'    : 6,
    'junqi'     : 0                ######这里要给它编一个号，要相对科学合理一些，用来计算炸弹利用率和剩余棋子大小
    }

def recognize_Hanzi(model, img, mode='red'):
    pre_result = None
    if mode == 'red':
        qizi_Hanzi,peaks = extract.extract_red(img)  # 提取红色棋子
    elif mode == 'green':
        qizi_Hanzi,peaks = extract.extract_green(img)  # 提取绿色棋子         ####extract是extract.py文件，之前import了一下下
    else:
        print('Error: mode must be red or green')
    
    if qizi_Hanzi is None:
        print('提取棋子失败')
    else:
        print('提取棋子成功')
        cv2.imshow('qizi_Hanzi', qizi_Hanzi)
        pre_result = tflite_predict.tflite_predict(model, qizi_Hanzi)  # 对图片中的文字进行预测 如要测试opencv部分性能，请将此句注释
    return pre_result,peaks


if __name__ == '__main__':
    if platform.system() == 'Windows':
        port = 'COM11'
    if platform.system() == 'Linux':
        port = '/dev/ttyUSB0' # 接树莓派左上USB口
    serialport = serial.Serial(port, 9600, timeout=0.5)
    if serialport.isOpen():
        print("open USART success")
    else:
        print("open USART failed")

    # 调试用
    # while bytes(serialport.readline()).decode('ascii') != 'Serial ON\r\n':
    #     print('wait...')
    # print('Serial ON')

    # time.sleep(5)
    # if serialport.isOpen():
    #     sendchar = 'A'
    #     # serialport.write(b'A\r\n')
    #     serialport.write(sendchar.encode('ascii'))
    #     receive_data = bytes(serialport.readline()).decode('ascii')
    #     if receive_data != '':
    #         print(receive_data[:-2])

    tflite_model_path = "./results/temp.tflite"  # 保存模型路径和名称
    if platform.system() == 'Windows':
        model = tf.lite.Interpreter(model_path = tflite_model_path) # Load TFLite model
    elif platform.system() == 'Linux':
        model = tflite.Interpreter(model_path = tflite_model_path)


    # img = cv2.imread('./origin_img/green/dilei/14.jpg')  # 读取图片
    # pre_result,_ = recognize_Hanzi(model, img, mode = 'green')
    # if pre_result is None:
    #     print('识别棋子失败')
    # else:
    #     print('识别棋子成功')
    #     print('pre_result:', pre_result)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    font = cv2.FONT_HERSHEY_SIMPLEX  # 设置字体样式
    cap = cv2.VideoCapture(0)          ####读取视频
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)      ####设置视频的长度和宽度
    fpsTime = time.time()
    last_save_time = fpsTime
    red_last_result = None
    green_last_result = None     #上一轮的结果
    red_remaining = 25
    green_remaining = 25       #剩余棋子数

    result_show={}         ######复盘结果储存在这里
    red_battledamage=green_battledamage=0        ######战损比
    red_remainsize=green_remainsize=0     
    red_totalremainsize=green_totalremainsize=111     ######剩余棋子大小统计
    red_kill=green_kill=0   ######击杀统计
    red_continuekill=green_continuekill=0  ######连续击杀统计
    red_largeuse=green_largeuse=0         ######大子利用率
    red_banguse=green_banguse=0         ######炸弹利用率
    battle_last_result='red_green'     ######记录上一轮结果，统计连续击杀时要用

    red_first=1
    green_first=1         ######棋盘上第一次少6枚棋子的记录，只有第一次少时，才会进行播报

    while cap.isOpened():
        _, frame = cap.read()
        # print(frame.shape)
        frame_green = frame[0:719,600:1279,:]
        frame_red = frame[0:719,0:700,:]

        green_result = None
        red_result = None
        compare_result = None
         
        green_pre_result,green_peaks = recognize_Hanzi(model, frame_green, mode = 'green') # 第一次识别
        if green_peaks is not None:
            # for peak in peaks:
            #     cv2.circle(frame, (int(peak[0]), int(peak[1])), 10, (0, 0, 255), -1) # 绘制顶点

            cv2.line(frame_green,(int(green_peaks[0][0]), int(green_peaks[0][1])),(int(green_peaks[1][0]), int(green_peaks[1][1])),(0,255,0),5,cv2.LINE_AA)
            cv2.line(frame_green,(int(green_peaks[1][0]), int(green_peaks[1][1])),(int(green_peaks[2][0]), int(green_peaks[2][1])),(0,255,0),5,cv2.LINE_AA)
            cv2.line(frame_green,(int(green_peaks[2][0]), int(green_peaks[2][1])),(int(green_peaks[3][0]), int(green_peaks[3][1])),(0,255,0),5,cv2.LINE_AA)
            cv2.line(frame_green,(int(green_peaks[3][0]), int(green_peaks[3][1])),(int(green_peaks[0][0]), int(green_peaks[0][1])),(0,255,0),5,cv2.LINE_AA)

        if green_pre_result is None:
            print('识别棋子失败')
            cv2.putText(frame,'Failed',(10, 500), font, 1,(0, 255, 0), 2, cv2.LINE_AA, 0)
        elif green_pre_result != green_last_result:
            print('识别棋子失败')
            cv2.putText(frame,'Failed',(10, 500), font, 1,(0, 255, 0), 2, cv2.LINE_AA, 0)
        else:
            print('识别棋子成功')
            cv2.putText(frame,'Success',(10, 500), font, 1,(0, 255, 0), 2, cv2.LINE_AA, 0)
            print('green_pre_result:', green_pre_result)
            cv2.putText(frame,'result:'+green_pre_result, (10, 550), font, 1,(0, 255, 0), 2, cv2.LINE_AA, 0)
            green_result = green_pre_result
            
        green_last_result = green_pre_result

        red_pre_result,red_peaks = recognize_Hanzi(model, frame_red, mode = 'red')
        if red_peaks is not None:
            # for peak in peaks:
            #     cv2.circle(frame, (int(peak[0]), int(peak[1])), 10, (0, 0, 255), -1) # 绘制顶点

            cv2.line(frame_red,(int(red_peaks[0][0]), int(red_peaks[0][1])),(int(red_peaks[1][0]), int(red_peaks[1][1])),(0,0,255),5,cv2.LINE_AA)
            cv2.line(frame_red,(int(red_peaks[1][0]), int(red_peaks[1][1])),(int(red_peaks[2][0]), int(red_peaks[2][1])),(0,0,255),5,cv2.LINE_AA)
            cv2.line(frame_red,(int(red_peaks[2][0]), int(red_peaks[2][1])),(int(red_peaks[3][0]), int(red_peaks[3][1])),(0,0,255),5,cv2.LINE_AA)
            cv2.line(frame_red,(int(red_peaks[3][0]), int(red_peaks[3][1])),(int(red_peaks[0][0]), int(red_peaks[0][1])),(0,0,255),5,cv2.LINE_AA)

        if red_pre_result is None:
            print('识别棋子失败')
            cv2.putText(frame,'Failed',(10, 100), font, 1,(0, 0, 255), 2, cv2.LINE_AA, 0)
        elif red_pre_result != red_last_result:
            print('识别棋子失败')
            cv2.putText(frame,'Failed',(10, 100), font, 1,(0, 0, 255), 2, cv2.LINE_AA, 0)
        else:
            print('识别棋子成功')
            cv2.putText(frame,'Success',(10, 100), font, 1,(0, 0, 255), 2, cv2.LINE_AA, 0)
            print('red_pre_result:', red_pre_result)
            cv2.putText(frame,'result:'+red_pre_result, (10, 150), font, 1,(0, 0, 255), 2, cv2.LINE_AA, 0)
            red_result = red_pre_result

        red_last_result = red_pre_result

        if red_result is not None and green_result is not None:


            ######加了个战损比battle_result,棋子利用率统计use_result
            compare_result,battle_result,use_result = compare.compare(red_result, green_result)   
            
            ######增加五项指标:战损比battledamage √、剩余棋子平均大小remainsize√、连续击杀(三次)continuekill √、大子利用率(个数)largeuse √、炸弹利用率(大小)banguse √
            
            result_show.append(red_result)             
            result_show.append(green_result)   
            result_show.append(compare_result)       ####result_show存储复盘结果，列表里每次按red_result、green_result、compare_result进行有序排列

            if battle_result=='red':
                green_battledamage=green_battledamage+1          ####计算战损比
                if use_result=='red_large':
                    red_largeuse=red_largeuse+1               ####计算大子利用率
            elif battle_result=='green':
                red_battledamage=red_battledamage+1
                if use_result=='green_large':
                    green_largeuse=green_largeuse+1
            elif battle_result=='red_green':
                red_battledamage=red_battledamage+1
                green_battledamage=green_battledamage+1                                
                if use_result=='red_bang':
                    if red_banguse==0:
                        red_bangnumber=1;                   
                    else:
                        red_bangnumber=2;   #需要判断一下下，这是第几次使用炸弹了
                    red_banguse=(red_banguse+piecenumber[green_result])/red_bangnumber     ####计算炸弹利用率，由于有两个，需平均
                elif use_result=='green_bang':
                    if green_banguse==0:
                        green_bangnumber=1;
                    else:
                        green_bangnumber=2;   #需要判断一下下，这是第几次使用炸弹了
                    green_banguse=(green_banguse+piecenumber[red_result])/green_bangnumber
            
            ######计算连续击杀
            if battle_result==battle_last_result:
                if battle_result=='red':
                    red_kill=red_kill+1
                if battle_result=='green':
                    green_kill=green_kill+1
            elif battle_result!=battle_last_result:
                red_kill=green_kill=0
            battle_last_result=battle_result
            if red_kill==3:
                red_continuekill=red_continuekill+1
                red_kill=0
            if green_kill==3:
                green_continuekill=green_continuekill+1
                green_kill=0
            
            ######计算剩余棋子平均大小
            red_totalremainsize=red_totalremainsize-piecenumber[red_result]
            green_totalremainsize=green_totalremainsize-piecenumber[green_result]
            red_remainsize=red_totalremainsize/(25-red_battledamage)
            green_remainsize=green_totalremainsize/(25-green_battledamage)

            print(compare_result)
            cv2.putText(frame,compare_result,(800, 50), font, 1,(255, 0, 0), 2, cv2.LINE_AA, 0)
            red_remaining, green_remaining = show_UI(compare_result, red_remaining, green_remaining) # UI显示
            ######
            
            # 将比较结果通过串口发送给Arduino
            if serialport.isOpen():
                sendchar = compareDict[compare_result]
                serialport.write(sendchar.encode('ascii'))
                receive_data = bytes(serialport.readline()).decode('ascii')
                if receive_data != '':
                    print(receive_data[:-2])
            # time.sleep(0.5) # 时间可调整

            #增加一个指令变量，用于语音模块播报提示
            if red_kill==3 and (red_remaining-green_remaining)==6 and green_first==1:
                sendletter='Q'
                green_first=0
            elif green_kill==3 and (green_remaining-red_remaining)==6 and red_first==1:
                sendletter='R'
                red_first=0
            elif red_kill==3:
                sendletter='M'
            elif green_kill==3:
                sendletter='N'
            elif (green_remaining-red_remaining)==6 and red_first==1:
                sendletter='O'
                red_first=0
            elif (red_remaining-green_remaining)==6 and green_first==1:
                sendletter='P'
                green_first=0
            # 将比较结果通过串口发送给Arduino
            if serialport.isOpen():
                sendchar = sendletter
                serialport.write(sendchar.encode('ascii'))
                receive_data = bytes(serialport.readline()).decode('ascii')
                if receive_data != '':
                    print(receive_data[:-2])
            # time.sleep(0.5) # 时间可调整
        
        if platform.system() == 'Windows':
            cv2.imshow('frame',frame)
        elif platform.system() == 'Linux':
            frame = cv2.resize(frame, (640, 480))
            cv2.imshow('frame',frame)

        cTime = time.time()
        fps_text = 1/(cTime-fpsTime)
        fpsTime = cTime
        cv2.putText(frame,str(round(fps_text,2))+'fps',(10, 50), font, 1,(255, 0, 0), 2, cv2.LINE_AA, 0)

        cv2.imshow('frame',frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
