import os
import random
import math
import shutil


def set_split(old_path, new_path, train_rate = 0.6, val_rate = 0.2):
    '''
    训练数据占train_rate
    验证数据占val_rate
    测试数据占1-train_rate-val_rate
    '''
    if os.path.exists(old_path) == 1: #文件夹存在，则新建一个新的文件夹
        if os.path.exists(new_path) != 1:
            os.makedirs(new_path)    #新建一个文件夹
    else:
        print('文件夹不存在！')
        return 1
    for path , sub_dirs , files in os.walk(old_path): #文件夹下三层文件，三级文件夹的路径
        for new_sub_dir in sub_dirs:
            filenames = os.listdir(os.path.join(path,new_sub_dir))   #filmenames 这时就是每个二级文件下 ，每张照片的名字

            filenames = list(filter(lambda x:x.endswith('.jpg') , filenames))   #把flimnames = x ,此时以.png结尾的文件通过过滤器 ，filter语法，后接函数还有序列 第一个为判断函数，第二个为序列
            
            random.shuffle(filenames) #把序列中所有元素，随机排序  得到一个打乱了的列表

            for i in range(len(filenames)):
                if i < math.floor(train_rate* len(filenames)):#math.floor  向下取整
                    sub_path = os.path.join(new_path , 'train_img',new_sub_dir)  #训练集
                elif i <math.floor((train_rate+val_rate)*len(filenames)):
                    sub_path = os.path.join(new_path , 'validation_img' , new_sub_dir)  #验证集
                else:
                    sub_path = os.path.join(new_path, 'test_img', new_sub_dir)   #测试集
                if os.path.exists(sub_path) == 0: #不存在时
                     os.makedirs(sub_path)  #新建一个文件夹
                shutil.copy(os.path.join(path, new_sub_dir,filenames[i]) , os.path.join(sub_path , filenames[i]))  #拷贝  从第一个路径拷贝到第二个路径下

if __name__ == '__main__':
    old_path = './extract_img'
    new_path = './data_img'
    set_split(old_path, new_path, train_rate = 0.72, val_rate = 0.22 )
