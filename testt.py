# -*- coding:utf-8 -*-
from Models import MyEndToEnd_Five
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint
from  Load_data import load_data,load_Img
import numpy as np

chars_Index = {0:"京",1: "沪",2: "津", 3:"渝", 4:"冀", 5:"晋",6:"蒙", 7:"辽", 8:"吉", 9:"黑", 10:"苏", 11:"浙", 12:"皖", 13:"闽", 14:"赣",
         15:"鲁", 16:"豫", 17:"鄂", 18:"湘", 19:"粤", 20:"桂",21:"琼", 22:"川",23: "贵",24: "云",25: "藏",26: "陕",27: "甘", 28:"青",
         29:"宁", 30:"新",31: "0", 32:"1",33: "2",34: "3",35: "4", 36:"5", 37:"6", 38:"7",39: "8", 40:"9",41: "A",42:"B",43: "C",
         44:"D",45: "E", 46:"F", 47:"G", 48:"H",49: "J", 50:"K", 51:"L", 52:"M",53: "N",54: "P",55: "Q",56: "R", 57:"S",58: "T",
         59:"U",60: "V", 61:"W",62: "X",63:"Y", 64:"Z" }

x_test = load_Img("testImg")
x_test = x_test.reshape(-1,72, 272,1).astype('float32')

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.0001,momentum=0.9, decay=0.0, nesterov=False)

Mymodel = MyEndToEnd_Five.MagicModel((72, 272, 1))
Mymodel.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
Mymodel.load_weights("NewR_Five_plate_best01.h5")

pre = Mymodel.predict(x_test)
pre = np.array(pre)
print(np.max(pre,axis=2))
pre = pre.argmax(axis=2)
print(pre)
pre_list = pre.tolist()
#for i in pre_list:
    #for j in i:
out=list(map(lambda x: [chars_Index[a] for a in list(x)], pre_list))
for i in out:
    print(i)


