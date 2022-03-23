#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
训练数据
author:Administrator
datetime:2018/3/24/024 19:52
software: PyCharm
'''
 

from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import SGD,Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

qizi = ['dilei','gongbin','junqi','junzhang','lianzhang','lvzhang','paizhang','shizhang','siling','tuanzhang','yinzhang','zhadan']
 
# 图片尺寸
img_width, img_height = 200, 100
input_shape = (img_width, img_height, 1)
 
train_data_dir = './data/train'
validation_data_dir = './data/validation'

# 图片生成器ImageDataGenerator
train_pic_gen = ImageDataGenerator(
    rescale=1. / 255,  # 对输入图片进行归一化到0-1区间
    # rotation_range=5,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    )
 
# 测试集不做变形处理，只需归一化。
validation_pic_gen = ImageDataGenerator(rescale=1. / 255)
 
# 按文件夹生成训练集流和标签，
train_flow = train_pic_gen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    color_mode='grayscale',
    # color_mode='rgb',
    # classes=qizi,
    classes=[str(i) for i in range(0,12)],
    class_mode='categorical')
 
# 按文件夹生成测试集流和标签，
validation_flow = validation_pic_gen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    color_mode='grayscale',
    # color_mode='rgb',
    # classes=qizi,
    classes=[str(i) for i in range(0,12)],
    class_mode='categorical'
)
 
 
# 搭建模型
# model = Sequential()

# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
 
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='softmax'))
# model.add(Dense(12))

model = Sequential([
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=2),

    # Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    # MaxPooling2D(pool_size=2),

    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),

    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),

    Flatten(),

    # Dense(64, activation='relu'),

    Dense(12, activation='softmax')
])


 
# sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00002), metrics=['accuracy'])


model.summary()


save_model_path = "results/temp.h5"  # 保存模型路径和名称

# lr_reduce = ReduceLROnPlateau('val_accuracy',patience=3,factor=0.1,min_lr=0.000001)


early_stop = EarlyStopping(monitor='val_accuracy',mode ='max', patience=5,verbose=1)

# 保存最佳训练参数
# checkpointer = ModelCheckpoint(filepath="./tmp/weights.hdf5", verbose=1, save_best_only=True)
checkpointer = ModelCheckpoint(filepath=save_model_path, monitor='val_accuracy',verbose=2,save_best_only=True,save_weights_only=False,mode='auto')

# 设置训练参数
nb_train_samples = 50
nb_validation_samples = 20
nb_epoch = 50


# 数据流训练API
# history = model.fit(
#     train_flow,
#     steps_per_epoch=nb_train_samples,
#     epochs=nb_epoch,
#     validation_data=validation_flow,
#     validation_steps=nb_validation_samples,
#     callbacks=[lr_reduce,checkpointer,early_stop]
#     )


history = model.fit(
    train_flow,
    steps_per_epoch=nb_train_samples,
    epochs=nb_epoch,
    validation_data=validation_flow,
    validation_steps=nb_validation_samples,
    callbacks=[checkpointer,early_stop]
    )

# print(history.history)

plt.figure(1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()


# model = load_model(save_model_path)

# model.save(save_model_path)