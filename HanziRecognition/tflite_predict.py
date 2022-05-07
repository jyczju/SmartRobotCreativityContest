import numpy as np
import cv2
import platform
if platform.system() == 'Windows':
    import tensorflow as tf
elif platform.system() == 'Linux':
    import tflite_runtime.interpreter as tflite

def tflite_predict(model, img):
    qizi = ['dilei', 'gongbin', 'junqi', 'junzhang', 'lianzhang', 'lvzhang',
        'paizhang', 'shizhang', 'siling', 'tuanzhang', 'yinzhang', 'zhadan']

    model.allocate_tensors()
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    img = img.astype(np.float32)
    input_data = img[np.newaxis, :, :, np.newaxis]

    index = input_details[0]['index']
    model.set_tensor(index, input_data)

    model.invoke() # 预测

    output_data = model.get_tensor(output_details[0]['index']) # 得到预测结果

    pre_class = np.argmax(output_data, axis=-1)
    pre_result = qizi[int(pre_class)]
    return pre_result

if __name__ == '__main__':
    tflite_model_path = "./results/temp.tflite"
    img_path = "./extract_img/junqi/ex_green_0.jpg"
    
    if platform.system() == 'Windows':
        model = tf.lite.Interpreter(model_path = tflite_model_path) # Load TFLite model
    elif platform.system() == 'Linux':
        model = tflite.Interpreter(model_path = tflite_model_path)
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    pre_result = tflite_predict(model, img)

    print(pre_result)

    cv2.waitKey(0)
