from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

from keras.preprocessing import image
import numpy as np

save_model_path = "results/temp.h5"  # 保存模型路径和名称
qizi = ['dilei','gongbin','junqi','junzhang','lianzhang','lvzhang','paizhang','shizhang','siling','tuanzhang','yinzhang','zhadan']

# 图片尺寸
img_width, img_height = 200, 100
input_shape = (img_width, img_height,1)
img_path = './qizi_data/shizhang.jpg'

img = image.load_img(img_path, grayscale=True, target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255
print(x.shape)

model = load_model(save_model_path)

# 进行预测,返回分类结果
classes = model.predict(x)
print(classes)

pre_class=np.argmax(classes,axis=-1)

pre_result=qizi[int(pre_class)]

print(pre_result)