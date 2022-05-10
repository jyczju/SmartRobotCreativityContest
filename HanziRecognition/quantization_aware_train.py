'''
jyc：仍有bug，显存被写爆
'''

# 实现h5文件转换为tflite文件，可以是fp32或int8，详解参见 https://zhuanlan.zhihu.com/p/165670135
import os
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
 
# 参数设置 ##############################################################################################

IMG_SIZE = (100, 150)
BATCH_SIZE = 32
q_epochs = 15             # 感知量化训练的轮数
learning_rate = 0.00001   # 感知量化学习率
num_test = 15             # 由于tflite在CPU上的推理速度有点慢，所以人为限制了最大数据量，其是设置为30就可以了，1000张图片
q_steps = 15             # 感知量化训练的步数
train_dir = "./data/train"
test_dir = "./data/validation"
weight_path = "./results/temp.h5"
output_path = "./results/"
qizi = ['dilei','gongbin','junqi','junzhang','lianzhang','lvzhang','paizhang','shizhang','siling','tuanzhang','yinzhang','zhadan']
 
 
# 读取数据和数据预处理

# 图片生成器ImageDataGenerator
train_pic_gen = ImageDataGenerator(
    rescale=1. / 255,  # 对输入图片进行归一化到0-1区间
    # rotation_range= 2,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    )
 
# 测试集不做变形处理，只需归一化。
test_pic_gen = ImageDataGenerator(rescale=1./ 255)

train_dataset = train_pic_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=32,
    color_mode='grayscale',
    classes=qizi,
    class_mode='categorical')

test_dataset = test_pic_gen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=32,
    color_mode='grayscale',
    classes=qizi,
    class_mode='categorical'
)
 
# 量化感知训练
 
# 读取模型并测试准确率
model = tf.keras.models.load_model(weight_path)
loss0, accuracy0 = model.evaluate(test_dataset)

# 量化感知训练，该量化感知训练应该是针对int8量化进行提前准备，如果想要为fp16量化提前准备，应该选择混合精度训练技术

with tf.device("/cpu:0"): # 使用cpu训练
# with tf.device("/gpu:0"):
    quantize_model = tfmot.quantization.keras.quantize_model
    q_aware_model = quantize_model(model)

    q_aware_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    q_aware_model.summary()
 
# 开始量化感知训练
    history_q = q_aware_model.fit(train_dataset, steps_per_epoch=q_steps, epochs=q_epochs)
 
# 测试输出模型准确率
loss1, accuracy1 = q_aware_model.evaluate(test_dataset)

# 量化感知模型转化为动态范围的tflite
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)      # 读取模型
converter.optimizations = [tf.lite.Optimize.DEFAULT]                     # 配置优化算法
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]   # 配置算子支持
# converter.target_spec.supported_types = [tf.int8]
quantized_tflite_model = converter.convert()                             # 转换模型
base_path = output_path + weight_path.split("/")[-1].rstrip(".h5")
output_model = "{}_int8.tflite".format(base_path)
with open(output_model, 'wb') as f:
    f.write(quantized_tflite_model)
 
# 量化感知模型转化为float32的tflite
# converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)      # 读取模型
# quantized_tflite_model = converter.convert()                             # 转换模型
# base_path = output_path + weight_path.split("/")[-1].rstrip(".h5")
# output_model = "{}_fp32.tflite".format(base_path) # , accuracy2)
# with open(output_model, 'wb') as f:
#     f.write(quantized_tflite_model)

# 定义int8量化所需数据
# def representative_dataset():
#     for images, _ in train_dataset.take(32):
#         for i in range(BATCH_SIZE):
#             image = np.expand_dims(images[i].numpy(), axis=0).astype(np.float32)
#             yield [image]

# def representative_dataset():
#     i = 0
#     for image in train_dataset:
#         i += 1
#         if i > 32:
#             break
#         img = np.expand_dims(image, axis=0).astype(np.float32)
#         yield [img]

 
# # 量化感知模型转化为int8的tflite 会报错
# converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)      # 读取模型
# converter.optimizations = [tf.lite.Optimize.DEFAULT]                     # 配置优化算法
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  # 配置算子支持
# converter.inference_input_type = tf.int8                                 # 设置输入数据为int8，如果不设置则默认fp32也就是说，输入fp32然后在网络里自己转换成int8
# converter.inference_output_type = tf.int8                                # 设置输出数据为int8，如果不设置则默认fp32也就是说，输入fp32然后在网络里自己转换成int8
# converter.representative_dataset = representative_dataset                # int8量化需要数据
# converter.allow_custom_ops = False
# converter.experimental_new_converter = True
# converter.experimental_new_quantizer = True
# quantized_tflite_model = converter.convert()
# base_path = output_path + weight_path.split("/")[-1].rstrip(".h5")
# output_model = "{}_int8.tflite".format(base_path)
# with open(output_model, 'wb') as f:
#     f.write(quantized_tflite_model)

print("h5 model: loss0, accuracy0 = {} {}".format(loss0, accuracy0))
print("tflite model: loss1, accuracy1 = {} {}".format(loss1, accuracy1))