import imp


import cv2
import extract


img = cv2.imread('./new_junqi/yinzhang/5.jpg')  # 读取图片
nameofpng = './qizi_data/yinzhang/5.jpg'
sourceImage = img.copy()  # 将原图做个备份

First_Hanzi = extract.recognize(img)
if First_Hanzi is None:
    print('提取棋子失败')
else:
    print('提取棋子成功')
    cv2.imshow('First_Hanzi', First_Hanzi)
    cv2.imwrite(nameofpng, First_Hanzi)

    

cv2.waitKey(0)
cv2.destroyAllWindows()