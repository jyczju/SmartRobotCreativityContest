import imp
from MSER_NMS import hanzi_box_detect
from predict_mser import predict_Hanzi
import cv2
from tensorflow.keras.models import load_model
import numpy as np


if __name__ == '__main__':
    save_model_path = "./results/temp_2class_986.h5"  # 保存模型路径和名称
    imagePath = r"origin_img\red\shizhang\9.jpg"

    model = load_model(save_model_path)
    img = cv2.imread(imagePath)
    orig = img.copy()

    hanzi_boxes = hanzi_box_detect(img)
    print("hanzi_boxes:")
    print(hanzi_boxes)

    max_result = 0
    max_x = 0
    max_y = 0
    max_x2 = 0
    max_y2 = 0
    for (x, y, x2, y2) in hanzi_boxes:
        cv2.rectangle(orig, (x, y), (x2, y2), (0, 255, 0), 2)
        img_tmp = img[y:y2, x:x2]
        h,w = img_tmp.shape[:2]
        gray = np.zeros((h,w), np.uint8)
        gray[:,:] = img_tmp[:,:,2] # 取红色通道
        gray = np.clip(gray, 0, 155)

        pre_result = predict_Hanzi(model, gray)
        print(pre_result)
        if pre_result > max_result:
            max_result = pre_result
            max_x = x
            max_y = y
            max_x2 = x2
            max_y2 = y2
    if max_result > 0.9:
        cv2.putText(orig, "qizi", (max_x, max_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow("hanzi_box", orig)
    cv2.waitKey(0)