import tensorflow as tf
from tensorflow.keras.models import load_model

h5_model_path =  "./results/temp.h5"
tflite_model_path = h5_model_path[:-2] + "tflite"
# print(tflite_model_path)

h5_model = load_model(h5_model_path)
converter = tf.lite.TFLiteConverter.from_keras_model(h5_model)

# 开启动态量化
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
open(tflite_model_path, "wb").write(tflite_model)
print("Convert h5 to tflite successfully!")