import tensorflow as tf
from tensorflow.keras.models import load_model

h5_model_path =  "./results/temp.h5"
tflite_model_path = h5_model_path[:-2] + "tflite"
# print(tflite_model_path)

h5_model = load_model(h5_model_path)
converter = tf.lite.TFLiteConverter.from_keras_model(h5_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT] # 开启动态量化
# converter.target_spec.supported_types = [tf.float32] # 开启支持32位浮点数
# converter.target_spec.supported_types = [tf.float16] # 开启支持16位浮点数


tflite_model = converter.convert()
open(tflite_model_path, "wb").write(tflite_model)
print("Convert h5 to tflite successfully!")