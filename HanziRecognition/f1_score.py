import numpy as np
import os
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

class F1_Score(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

        qizi = ['dilei','gongbin','junqi','junzhang','lianzhang','lvzhang','paizhang','shizhang','siling','tuanzhang','yinzhang','zhadan']
        files_num = 0
        for i in range(0,12):
            img_dir = './data/validation/' + qizi[i]
            for _,_,files in os.walk(img_dir):
                pass
            files_num += len(files)
        print('validation_data has '+str(int(files_num))+' files.')

        validation_pic_gen = ImageDataGenerator(rescale=1. / 255)
        validation_flow = validation_pic_gen.flow_from_directory(
            './data/validation',
            target_size=(100, 200),
            batch_size=files_num, # 32
            color_mode='grayscale',
            classes=['dilei','gongbin','junqi','junzhang','lianzhang','lvzhang','paizhang','shizhang','siling','tuanzhang','yinzhang','zhadan'],
            class_mode='categorical'
        )
        self.validation_data = validation_flow

    def on_epoch_end(self, epoch, logs={}):
#         val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
#         val_targ = self.validation_data[1]
        # print(len(self.validation_data))
        # print(logs)
        val_predict = []
        val_targ = []
        for i in range(0,len(self.validation_data)):
             val_predict = np.hstack([val_predict,np.argmax(np.asarray(self.model.predict(self.validation_data[i][0])), axis=1)])
             val_targ = np.hstack([val_targ,np.argmax(self.validation_data[i][1], axis=1)])
        # print(val_predict)
        # print(val_targ)
        # val_predict = np.argmax(np.asarray(self.model.predict(self.validation_data[0][0])), axis=1)

        # val_targ = np.argmax(self.validation_data[0][1], axis=1)
        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        # _val_recall = recall_score(val_targ, val_predict)
        # _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        # self.val_recalls.append(_val_recall)
        # self.val_precisions.append(_val_precision)
        # print(' - val_f1: %f - val_precision: %f - val_recall %f' %(_val_f1, _val_precision, _val_recall))
        print(' - val_f1:' ,_val_f1)
        # logs['val_f1'] = _val_f1
        # logs.update(logs)
        # print(logs)
        # return super().on_epoch_end(epoch, logs)
        return

    def on_train_end(self, logs=None):
        plt.figure(0)
        x = range(0,len(self.val_f1s))
        y = [i for i in self.val_f1s]
        plt.plot(x,y, label="val_f1")
        plt.title('val_f1')
        plt.legend()
        plt.show()
        return super().on_train_end(logs)


# 其他metrics可自行添加

if __name__ == '__main__':
    f1_score = F1_Score()
