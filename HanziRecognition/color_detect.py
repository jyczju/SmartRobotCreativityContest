from MSER_NMS import hanzi_box_detect
from predict_mser import predict_Hanzi
import cv2


if __name__ == '__main__':
    # 读取图片
    imagePath = r"origin_img\red\zhadan\14.jpg"
    img = cv2.imread(imagePath)
    orig = img.copy()

    hanzi_boxes = hanzi_box_detect(img)
    print(hanzi_boxes)
    for (x, y, x2, y2) in hanzi_boxes:
        cv2.rectangle(orig, (x, y), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("hanzi_box", orig)

    


    # color = color_detect(img)
    # # print("color:", color)
    # h,s,v = rgb2hsv(color[2], color[1], color[0])
    # print("hsv:", h,s,v)

    cv2.waitKey(0)