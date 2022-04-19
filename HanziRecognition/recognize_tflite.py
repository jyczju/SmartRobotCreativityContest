import cv2
import extract
import tflite_predict
import tensorflow as tf
import time

def recognize_Hanzi(model, img, mode='red'):
    pre_result = None
    if mode == 'red':
        qizi_Hanzi,peaks = extract.extract_red(img)  # 提取红色棋子
    elif mode == 'green':
        qizi_Hanzi,peaks = extract.extract_green(img)  # 提取绿色棋子
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

    tflite_model_path = "./results/temp.tflite"  # 保存模型路径和名称
    model = tf.lite.Interpreter(model_path = tflite_model_path) # Load TFLite model


    # img = cv2.imread('./new_junqi/yinzhang/5.jpg')  # 读取图片
    # pre_result = recognize_Hanzi(model, img, mode = 'green')
    # if pre_result is None:
    #     print('识别棋子失败')
    # else:
    #     print('识别棋子成功')
    #     print('pre_result:', pre_result)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



    font = cv2.FONT_HERSHEY_SIMPLEX  # 设置字体样式
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fpsTime = time.time()
    last_save_time = fpsTime
    last_result = None
    while cap.isOpened():
        _, frame = cap.read()
        # print(frame.shape)
         
        pre_result,peaks = recognize_Hanzi(model, frame,mode = 'green') # 第一次识别
        if peaks is not None:
            # for peak in peaks:
            #     cv2.circle(frame, (int(peak[0]), int(peak[1])), 10, (0, 0, 255), -1) # 绘制顶点

            cv2.line(frame,(int(peaks[0][0]), int(peaks[0][1])),(int(peaks[1][0]), int(peaks[1][1])),(0,0,255),5,cv2.LINE_AA)
            cv2.line(frame,(int(peaks[1][0]), int(peaks[1][1])),(int(peaks[2][0]), int(peaks[2][1])),(0,0,255),5,cv2.LINE_AA)
            cv2.line(frame,(int(peaks[2][0]), int(peaks[2][1])),(int(peaks[3][0]), int(peaks[3][1])),(0,0,255),5,cv2.LINE_AA)
            cv2.line(frame,(int(peaks[3][0]), int(peaks[3][1])),(int(peaks[0][0]), int(peaks[0][1])),(0,0,255),5,cv2.LINE_AA)

        if pre_result is None:
            print('识别棋子失败')
            cv2.putText(frame,'Failed',(10, 100), font, 1,(0, 0, 255), 2, cv2.LINE_AA, 0)
        elif pre_result != last_result:
            print('识别棋子失败')
            cv2.putText(frame,'Failed',(10, 100), font, 1,(0, 0, 255), 2, cv2.LINE_AA, 0)
        else:
            print('识别棋子成功')
            cv2.putText(frame,'Success',(10, 100), font, 1,(0, 0, 255), 2, cv2.LINE_AA, 0)
            print('pre_result:', pre_result)
            cv2.putText(frame,'result:'+pre_result, (10, 150), font, 1,(0, 0, 255), 2, cv2.LINE_AA, 0)
            

        last_result = pre_result

        cTime = time.time()
        fps_text = 1/(cTime-fpsTime)
        fpsTime = cTime
        cv2.putText(frame,str(int(fps_text))+'fps',(10, 50), font, 1,(0, 0, 255), 2, cv2.LINE_AA, 0)

        cv2.imshow('frame',frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

