from tensorflow.keras.models import load_model
import numpy as np
import cv2


def predict_Hanzi(model, img):
    '''判断图片中是否含有棋子'''
    h,w = img.shape[:2]
    # print("h,w:",h,w)
    if h != 100 or w != 150:
        if w > h:
            img = cv2.resize(img, (150, 100))
        else:           
            img = cv2.resize(img, (100, 150))
            img = np.rot90(img)
            # print(img.shape)
    
    qizi = ['qizi', 'other']
    img = img.astype(float)
    x = img[np.newaxis, :, :, np.newaxis]

    x /= 255
    # print(x.shape)

    # 进行预测,返回分类结果
    classes = model.predict(x)
    # print(classes) # 输出概率值

    # pre_class = np.argmax(classes, axis=-1)
    # print(pre_class)

    # pre_result = qizi[int(pre_class)]
    return classes[0][0]


if __name__ == '__main__':
    save_model_path = "./results/temp_2class_960.h5"  # 保存模型路径和名称
    # img_path = './data_mser/test/qizi/qizi_0_39.jpg'
    img_path = r'data_img\test_img\dilei\ex_green_14.jpg'

    model = load_model(save_model_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 读取图片

    pre_result = predict_Hanzi(model, img)  # 对图片中的文字进行预测

    print(pre_result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
