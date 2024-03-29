#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
生成多个样本
author:Administrator
datetime:2018/3/24/024 18:46
software: PyCharm
'''
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
import os
 
 
# 生成文件夹
def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError:
            pass
 
 
# 图片生成器ImageDataGenerator
pic_gen = ImageDataGenerator(
    rotation_range=5, 
    width_shift_range=0, # 0
    height_shift_range=0, # 0
    shear_range=0.1, # 0.2
    zoom_range=0.1, # 0.2
    fill_mode='nearest')
 
# 生成图片
def img_create(img_dir, save_dir, img_prefix, num=100):
    img = load_img(img_dir)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    img_flow = pic_gen.flow(
        x,
        batch_size=1,
        save_to_dir=save_dir,
        save_prefix=img_prefix,
        save_format="jpg"
    )
    i = 0
    for batch in img_flow:
        i += 1
        if i > num:
            break
 

qizi = ['dilei','gongbin','junqi','junzhang','lianzhang','lvzhang','paizhang','shizhang','siling','tuanzhang','yinzhang','zhadan']

# 生成训练集
for i in range(0, 12):
    save_dir = './data/train/' + qizi[i] #str(i) # 保存文件夹
    img_dir = './data_img/train_img/' + qizi[i] # 来源文件夹
    for _, _, files in os.walk(img_dir):
        # 遍历文件
        for f in files:
            img_file_dir = img_dir + '/' + f
            ensure_dir(save_dir)
            img_create(img_file_dir, save_dir, qizi[i], num=45)
    print("train: ", i)

# 生成验证集
for i in range(0, 12):
    save_dir = './data/validation/' + qizi[i] #str(i)
    img_dir = './data_img/validation_img/' + qizi[i]
    for _, _, files in os.walk(img_dir):
        # 遍历文件
        for f in files:
            img_file_dir = img_dir + '/' + f
            ensure_dir(save_dir)
            img_create(img_file_dir, save_dir, qizi[i], num=15)
    print("validation: ", i)

# 生成测试集
for i in range(0, 12):
    save_dir = './data/test/' + qizi[i] # 保存文件夹
    img_dir = './data_img/test_img/' + qizi[i] # 来源文件夹
    for _, _, files in os.walk(img_dir):
        # 遍历文件
        for f in files:
            img_file_dir = img_dir + '/' + f # 图片文件路径
            ensure_dir(save_dir)
            img_create(img_file_dir, save_dir, qizi[i], num=3)
    print("test: ", i)


