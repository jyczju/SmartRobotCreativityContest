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
import matplotlib.pyplot as plt
 
# 参数设置

img_size = (100, 150)
batch_size = 32
q_epochs = 20          # 感知量化训练的轮数
learning_rate = 0.00001   # 感知量化学习率
q_steps = 64            # 感知量化训练的步数
train_dir = "./data/train"
test_dir = "./data/test"
weight_path = "./results/temp.h5"
output_path = "./results/"
qizi = ['dilei','gongbin','junqi','junzhang','lianzhang','lvzhang','paizhang','shizhang','siling','tuanzhang','yinzhang','zhadan']
 
 
# 读取数据和数据预处理
#生成图片与对应标签的字典
def load_sample(sample_dir):
    #图片名列表
    lfilenames = []
    #标签名列表
    labelnames = []
    #遍历文件夹
    for (dirpath,dirnames,filenames) in os.walk(sample_dir):
        #遍历图片
        for filename in filenames:
            #每张图片的路径名
            filename_path = os.sep.join([dirpath,filename])
            #添加文件名
            lfilenames.append(filename_path)
            #添加文件名对应的标签
            labelnames.append(dirpath.split('/')[-1])
            
    #生成标签名列表
    lab = list(sorted(set(labelnames)))
    #生成标签字典
    labdict = dict(zip(lab,list(range(len(lab)))))
    #生成与图片对应的标签列表
    labels = [labdict[i] for i in labelnames]
    #图片与标签字典
    image_label_dict = dict(zip(lfilenames,labels))
    #将文件名与标签列表打乱
    lfilenames = []
    labels = []
    for key in image_label_dict:
        lfilenames.append(key)
        labels.append(image_label_dict[key])
    #返回文件名与标签列表
    return lfilenames,labels

train_filenames,train_labels = load_sample(train_dir)
test_filenames,test_labels = load_sample(test_dir)
# print(train_filenames,train_labels)

#将图片制成Dataset方法
def make_Dataset(filenames,labels,size,batch_size):
    #生成dataset对象
    dataset = tf.data.Dataset.from_tensor_slices((filenames,labels))
    #转化为图片数据集
    dataset = dataset.map(_parseone)
    #按批次组合数据
    dataset = dataset.batch(batch_size)
    return dataset

#解析图片文件的方法
def _parseone(filename,label):
    #读取所有图片
    image_string = tf.io.read_file(filename)
    #将图片解码并返回空的shape
    image_decoded = tf.image.decode_image(image_string, channels=1)
    #因为是空的shape，所以需要设置shape
    image_decoded.set_shape([None,None,None])
    image_decoded = tf.image.resize(image_decoded,(100,150))
    # print(image_decoded.shape)
    #归一化
    image_decoded = image_decoded/255.
    #将归一化后的像素矩阵转化为image张量
    image_decoded = tf.cast(image_decoded,dtype=tf.float32)
    #将label转为张量
    label = tf.cast(tf.reshape(label,[]),dtype=tf.int32)
    #将标签制成one_hot
    label = tf.one_hot(label,depth=classes_num,on_value=1)
    return image_decoded,label

#从Dataset中取出数据的方法
def getdata(dataset):
    #生成一个迭代器
    iterator = dataset.make_one_shot_iterator()
    #从迭代器中取出一个数据
    one_element = iterator.get_next()
    return one_element

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

#设置参数
#训练集与测试集总的数量
train_total = len(train_labels)
test_total = len(test_labels)
#图片总共由12类，用于one_hot标签
classes_num = 12
#将所有图片训练一轮所需要的总的训练次数
total_batch = int(train_total/batch_size) 

#制作训练与测试集Dataset
train_dataset = make_Dataset(train_filenames,train_labels,img_size,batch_size)
test_dataset = make_Dataset(test_filenames,test_labels,img_size,test_total)

 
# 量化感知训练
# 读取模型并测试准确率
model = tf.keras.models.load_model(weight_path)
# loss0, accuracy0 = model.evaluate(test_dataset)

# 量化感知训练，该量化感知训练应该是针对int8量化进行提前准备，如果想要为fp16量化提前准备，应该选择混合精度训练技术

# with tf.device("/cpu:0"): # 使用cpu训练
with tf.device("/gpu:0"):
    quantize_model = tfmot.quantization.keras.quantize_model
    q_aware_model = quantize_model(model)

    q_aware_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    q_aware_model.summary()
 
# 开始量化感知训练
    history_q = q_aware_model.fit(train_dataset, steps_per_epoch=q_steps, epochs=q_epochs) #, validation_data=test_dataset, validation_steps=test_total)


plt.figure(1)
plt.subplot(121)
plt.plot(history_q.history['loss'], label='train_loss')
# plt.plot(history_q.history['val_loss'], label='val_loss')
plt.title('loss')
plt.legend()

plt.subplot(122)
plt.plot(history_q.history['accuracy'], label='train_accuracy')
# plt.plot(history_q.history['val_accuracy'], label='val_accuracy')
plt.title('accuracy')
plt.legend()

plt.savefig('history.png')
plt.show()
 
# 测试输出模型准确率
# loss1, accuracy1 = q_aware_model.evaluate(test_dataset)

# 量化感知模型转化为动态范围的tflite
# converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)      # 读取模型
# converter.optimizations = [tf.lite.Optimize.DEFAULT]                     # 配置优化算法
# # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS] # 配置算子支持
# # converter.target_spec.supported_types = [tf.int8]
# quantized_tflite_model = converter.convert()                             # 转换模型
# base_path = output_path + weight_path.split("/")[-1].rstrip(".h5")
# output_model = "{}_int8.tflite".format(base_path)
# with open(output_model, 'wb') as f:
#     f.write(quantized_tflite_model)
 
# 量化感知模型转化为float32的tflite
# converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)      # 读取模型
# quantized_tflite_model = converter.convert()                             # 转换模型
# base_path = output_path + weight_path.split("/")[-1].rstrip(".h5")
# output_model = "{}_fp32.tflite".format(base_path) # , accuracy2)
# with open(output_model, 'wb') as f:
#     f.write(quantized_tflite_model)

# 定义int8量化所需数据
def representative_dataset():
    for images, _ in train_dataset.take(32):
        for i in range(batch_size):
            image = np.expand_dims(images[i].numpy(), axis=0).astype(np.float32)
            yield [image]

# 量化感知模型转化为int8的tflite 会报错
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)      # 读取模型
converter.optimizations = [tf.lite.Optimize.DEFAULT]                     # 配置优化算法
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  # 配置算子支持
# converter.inference_input_type = tf.int8                                 # 设置输入数据为int8，如果不设置则默认fp32也就是说，输入fp32然后在网络里自己转换成int8
# converter.inference_output_type = tf.int8                                # 设置输出数据为int8，如果不设置则默认fp32也就是说，输入fp32然后在网络里自己转换成int8
converter.representative_dataset = representative_dataset                # int8量化需要数据
converter.allow_custom_ops = False
converter.experimental_new_converter = True
converter.experimental_new_quantizer = True
# converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,tf.lite.OpsSet.TFLITE_BUILTINS]
quantized_tflite_model = converter.convert()
print("Start to convert")
base_path = output_path + weight_path.split("/")[-1].rstrip(".h5")
output_model = "{}_int8.tflite".format(base_path)
with open(output_model, 'wb') as f:
    f.write(quantized_tflite_model)

# print("h5 model: loss0, accuracy0 = {} {}".format(loss0, accuracy0))
# print("tflite model: loss1, accuracy1 = {} {}".format(loss1, accuracy1))

print("Convert h5 to tflite successfully!")