from cmath import inf
from MSER_NMS import hanzi_box_detect,rgb2hsv
from predict_mser import predict_Hanzi
import cv2
from tensorflow.keras.models import load_model
import numpy as np

def color_hsv_detect(img, model, mode = 'red'):
    color_hsv = [0,0,0]
    
    if mode == 'red':
        img = img[0:719,0:700,:]
    elif mode == 'green':
        img = img[0:719,600:1279,:]
    orig = img.copy()

    hanzi_boxes = hanzi_box_detect(img, mode)
    # print("hanzi_boxes:")
    # print(hanzi_boxes)

    max_result = 0
    max_x = 0
    max_y = 0
    max_x2 = 0
    max_y2 = 0
    for (x, y, x2, y2) in hanzi_boxes:
        cv2.rectangle(orig, (x, y), (x2, y2), (0, 255, 0), 2)
        img_tmp = img[y:y2, x:x2]
        gray = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2GRAY)
        # gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imshow("gray", gray)

        pre_result = predict_Hanzi(model, gray)
        # print(pre_result)
        if pre_result > max_result:
            max_result = pre_result
            max_x = x
            max_y = y
            max_x2 = x2
            max_y2 = y2

    if max_result > 0.9:
        cv2.putText(orig, "qizi", (max_x, max_y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        center = (int((max_x + max_x2) / 2), int((max_y + max_y2) / 2))
        cv2.circle(orig, center, 5, (0, 0, 255), -1)
        cv2.rectangle(orig, (max_x, max_y), (max_x2, max_y2), (0, 0, 255), 2)

        # sum = 0
        # color = [0,0,0]
        # for i in range(center[0]-20, center[0]+20):
        #     for j in range(center[1]-20, center[1]+20):
        #         light = 0.114*int(orig[i,j,0]) + 0.587*int(orig[i,j,1]) + 0.299*int(orig[i,j,2])
        #         if light > 100:
        #             sum += 1
        #             color[2] += orig[i,j,0]
        #             color[1] += orig[i,j,1]
        #             color[0] += orig[i,j,2]
        # color[0] /= sum # red
        # color[1] /= sum # green
        # color[2] /= sum # blue
        # color_hsv = rgb2hsv(color[0], color[1], color[2])

        sum = 0
        color_hsv = [0,0,0]
        max_hsv = [-inf,-inf ,-inf]
        for i in range(center[0]-20, center[0]+20):
            for j in range(center[1]-20, center[1]+20):
                light = 0.114*int(orig[i,j,0]) + 0.587*int(orig[i,j,1]) + 0.299*int(orig[i,j,2])
                if light > 100:
                    sum += 1
                    h1,s1,v1 = rgb2hsv(orig[i,j,2], orig[i,j,1], orig[i,j,0])
                    color_hsv[0] += h1
                    color_hsv[1] += s1
                    color_hsv[2] += v1
                    if h1 > max_hsv[0]:
                        max_hsv[0] = h1
                    if s1 > max_hsv[1]:
                        max_hsv[1] = s1
                    if v1 > max_hsv[2]:
                        max_hsv[2] = v1
        color_hsv[0] = int(color_hsv[0]/sum) # red
        color_hsv[1] = int(color_hsv[1]/sum) # green
        color_hsv[2] = int(color_hsv[2]/sum) # blue

        cv2.putText(orig, "hsv_low: " + str(int(color_hsv[0]-3))+','+str(int(color_hsv[1]-130))+','+str(int(color_hsv[2]-180)), (max_x, max_y2+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(orig, "hsv_high: " + str(int(max_hsv[0]+9))+','+str(int(max_hsv[1]+50))+','+str(int(max_hsv[2]+20)), (max_x, max_y2+60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("hanzi_box", orig)
    return np.array(color_hsv,dtype=int), np.array(max_hsv,dtype=int)

if __name__ == '__main__':
    save_model_path = "./results/temp_2class_960.h5"  # 保存模型路径和名称
    model = load_model(save_model_path)
    
    imagePath = r"origin_img\red\lvzhang\WIN_20220507_08_17_30_Pro.jpg"
    img = cv2.imread(imagePath)
    color_hsv,max_hsv = color_hsv_detect(img, model)
    print(color_hsv,max_hsv)
    
    cv2.waitKey(0)

# if __name__ == '__main__':
#     save_model_path = "./results/temp_2class_960.h5"  # 保存模型路径和名称
#     model = load_model(save_model_path)
    
#     imagePath = r"origin_img\green\zhadan\WIN_20220507_08_36_52_Pro.jpg"
#     img = cv2.imread(imagePath)
#     color_hsv = color_hsv_detect(img, model, mode = 'green')
#     print(color_hsv)
    
#     cv2.waitKey(0)