import os
import tensorflow as tf
import cv2
from tflite_predict import tflite_predict
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import numpy as np

def evaluate(model, test_path):
    '''评价模型对各棋子分类的准确率'''
    print('Accuracy of tflite model:')
    test_predict = []
    test_targ = []
    qizi = ['dilei','gongbin','junqi','junzhang','lianzhang','lvzhang','paizhang','shizhang','siling','tuanzhang','yinzhang','zhadan']
    files_num = 0
    right_nums = 0
    for i in range(0,12):
        img_dir = test_path + '/' + qizi[i]
        right_num = 0
        for _,_,files in os.walk(img_dir):
            # 遍历文件
            for f in files:
                # print('evaluating by ',f)

                img_file_dir = img_dir + '/' + f
                img = cv2.imread(img_file_dir, cv2.IMREAD_GRAYSCALE)  # 读取图片

                pre_result = tflite_predict(model, img)  # 对图片中的文字进行预测

                test_predict = np.hstack([test_predict, pre_result])
                test_targ = np.hstack([test_targ, qizi[i]])

                if pre_result == qizi[i]:
                    right_num += 1

        right_rate = float(right_num)/len(files)
        files_num += len(files)
        right_nums += right_num
        print(qizi[i],':',right_rate)

    _val_f1 = f1_score(test_targ, test_predict, average='micro') # average='macro'
    print('f1_score:' ,_val_f1)
    acc = float(right_nums)/files_num
    print('accuracy:', acc)

if __name__ == '__main__':
    tflite_model_path = "./results/temp_int8.tflite"
    test_path = './data/test'
    model = tf.lite.Interpreter(model_path = tflite_model_path) # Load TFLite model
    
    evaluate(model, test_path)




