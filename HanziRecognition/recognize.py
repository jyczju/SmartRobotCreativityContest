import cv2
import extract
import predict
from tensorflow.keras.models import load_model
import time

def recognize_Hanzi(model, img, mode='red'):
    pre_result = None
    if mode == 'red':
        qizi_Hanzi = extract.extract_red(img)  # 提取红色棋子
    elif mode == 'green':
        qizi_Hanzi = extract.extract_green(img)  # 提取绿色棋子
    else:
        print('Error: mode must be red or green')
    
    if qizi_Hanzi is None:
        print('提取棋子失败')
    else:
        print('提取棋子成功')
        cv2.imshow('qizi_Hanzi', qizi_Hanzi)
        pre_result = predict.predict_Hanzi(model, qizi_Hanzi)  # 对图片中的文字进行预测 如要测试opencv部分性能，请将此句注释
    return pre_result


if __name__ == '__main__':

    save_model_path = "results/Best2CNN100.h5"  # 保存模型路径和名称
    model = load_model(save_model_path)



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
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fpsTime = time.time()
    last_save_time = fpsTime
    last_result = None
    while cap.isOpened():
        _, frame = cap.read()
        # print(frame.shape)
         
        pre_result = recognize_Hanzi(model, frame,mode = 'green') # 第一次识别

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

