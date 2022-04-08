import cv2
import predict
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf

# def evaluate(model, test_path):
#     '''评价模型对各棋子分类的准确率'''
#     print('Accuracy of model for qizi:')
#     qizi = ['dilei','gongbin','junqi','junzhang','lianzhang','lvzhang','paizhang','shizhang','siling','tuanzhang','yinzhang','zhadan']
#     files_num = 0
#     for i in range(0,12):
#         img_dir = test_path + '/' + qizi[i]
#         right_num = 0
#         for _,_,files in os.walk(img_dir):
#             # 遍历文件
#             for f in files:
#                 img_file_dir = img_dir + '/' + f
#                 img = cv2.imread(img_file_dir, cv2.IMREAD_GRAYSCALE)  # 读取图片

#                 pre_result = predict.predict_Hanzi(model, img)  # 对图片中的文字进行预测
#                 if pre_result == qizi[i]:
#                     right_num += 1
#                 # print(pre_result)
#         right_rate = float(right_num)/len(files)
#         files_num += len(files)
#         print(qizi[i],':',right_rate)
    
#     print('test_data has '+str(int(files_num))+' files.')

#     test_pic_gen = ImageDataGenerator(rescale=1. / 255)
#     test_flow = test_pic_gen.flow_from_directory(
#         test_path,
#         target_size=(100, 200),
#         batch_size=files_num,
#         color_mode='grayscale',
#         classes=['dilei','gongbin','junqi','junzhang','lianzhang','lvzhang','paizhang','shizhang','siling','tuanzhang','yinzhang','zhadan'],
#         class_mode='categorical'
#     )
#     test_predict = []
#     test_targ = []
#     for i in range(0,len(test_flow)):
#         test_predict = np.hstack([test_predict,np.argmax(np.asarray(model.predict(test_flow[i][0])), axis=1)])
#         test_targ = np.hstack([test_targ,np.argmax(test_flow[i][1], axis=1)])
#     _val_f1 = f1_score(test_targ, test_predict, average='macro')
#     print('f1_score:' ,_val_f1)

def evaluate(model, test_path):
    '''评价模型对各棋子分类的准确率'''
    print('Accuracy of model for qizi:')
    test_predict = []
    test_targ = []
    qizi = ['dilei','gongbin','junqi','junzhang','lianzhang','lvzhang','paizhang','shizhang','siling','tuanzhang','yinzhang','zhadan']
    # files_num = 0
    for i in range(0,12):
        img_dir = test_path + '/' + qizi[i]
        right_num = 0
        for _,_,files in os.walk(img_dir):
            # 遍历文件
            for f in files:
                img_file_dir = img_dir + '/' + f
                img = cv2.imread(img_file_dir, cv2.IMREAD_GRAYSCALE)  # 读取图片

                pre_result = predict.predict_Hanzi(model, img)  # 对图片中的文字进行预测
                
                test_predict = np.hstack([test_predict, pre_result]) # 预测结果
                test_targ = np.hstack([test_targ, qizi[i]]) # 标签
                
                if pre_result == qizi[i]:
                    right_num += 1

        right_rate = float(right_num)/len(files)
        # files_num += len(files)
        print(qizi[i],':',right_rate)

    _val_f1 = f1_score(test_targ, test_predict, average='micro') # average='macro' # 计算f1_score
    print('f1_score:' ,_val_f1)

if __name__ == '__main__':
    save_model_path = "results/temp868Best.h5"  # 保存模型路径和名称
    test_path = './data/test'
    model = load_model(save_model_path)
    
    evaluate(model, test_path)




