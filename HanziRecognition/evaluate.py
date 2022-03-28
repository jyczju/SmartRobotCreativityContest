import cv2
import predict
import os
from tensorflow.keras.models import load_model

def evaluate(model, test_path):
    '''评价模型对各棋子分类的准确率'''
    print('Accuracy of model for qizi:')
    qizi = ['dilei','gongbin','junqi','junzhang','lianzhang','lvzhang','paizhang','shizhang','siling','tuanzhang','yinzhang','zhadan']
    for i in range(0,12):
        img_dir = test_path + '/' + qizi[i]
        right_num = 0
        for _,_,files in os.walk(img_dir):
            # 遍历文件
            for f in files:
                img_file_dir = img_dir + '/' + f
                img = cv2.imread(img_file_dir, cv2.IMREAD_GRAYSCALE)  # 读取图片

                pre_result = predict.predict_Hanzi(model, img)  # 对图片中的文字进行预测
                if pre_result == qizi[i]:
                    right_num += 1
                # print(pre_result)
        right_rate = float(right_num)/len(files)
        print(qizi[i],':',right_rate)

if __name__ == '__main__':
    save_model_path = "results/temp806.h5"  # 保存模型路径和名称
    model = load_model(save_model_path)
    test_path = './data/test'

    evaluate(model, test_path)


