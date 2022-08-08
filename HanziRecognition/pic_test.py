import cv2
import numpy as np

if __name__ =="__main__":
    img = cv2.imread("./origin_img/red/junzhang/0.jpg")
    h,w = img.shape[:2]
    gray = np.zeros((h,w), np.uint8)
    gray[:,:] = img[:,:,2] # 取红色通道
    gray = np.clip(gray, 0, 155)
    cv2.imshow("img", img)
    cv2.imshow("gray", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()