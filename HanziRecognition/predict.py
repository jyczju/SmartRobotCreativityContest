from pickletools import uint8
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
from PIL import Image

save_model_path = "results/temp.h5"  # 保存模型路径和名称
qizi = ['dilei','gongbin','junqi','junzhang','lianzhang','lvzhang','paizhang','shizhang','siling','tuanzhang','yinzhang','zhadan']

# 图片尺寸
height, width = 100, 200
# img_path = './test/test1.jpg'
img_path = './qizi_data/dilei/0.jpg'


img_cv = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)  # 读取图片
# print(img_cv)
# cv2.imshow('img_cv',img_cv)
img_cv = img_cv.astype(float)
x = img_cv[np.newaxis,:,:,np.newaxis]

x /= 255
# print(x.shape)

model = load_model(save_model_path)

# 进行预测,返回分类结果
classes = model.predict(x)
# print(classes) # 输出概率值

pre_class=np.argmax(classes,axis=-1)

pre_result=qizi[int(pre_class)]

print(pre_result)

cv2.waitKey(0)
cv2.destroyAllWindows()