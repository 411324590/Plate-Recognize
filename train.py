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

#x_train,y_train =load_data("data/plate_train")
x_test,y_test =load_data("data/plate_test")
# x_test = load_Img("testImg")
#x_train = x_train.reshape(-1, 72, 272,1).astype('float32')
x_test = x_test.reshape(-1,72, 272,1).astype('float32')

# mnist = input_data.read_data_sets(".\\MNIST_data", one_hot=True)
# x_train, y_train = mnist.train.images,mnist.train.labels
# x_test, y_test = mnist.test.images, mnist.test.labels
# x_train = x_train.reshape(-1, 28, 28,1).astype('float32')
# x_test = x_test.reshape(-1,28, 28,1).astype('float32')
# G = GenPlate("./font/platech.ttf",'./font/platechar.ttf',"./NoPlates")
# def gen_data(batch_size=64):
#     while True:
#         l_plateStr, l_plateImg = G.genBatch(batch_size, 2, range(31, 65), "plateImg", (272, 72))
#         X = np.array(l_plateImg, dtype=np.uint8)/255
#         X = X.reshape(-1, 72, 272, 1).astype('float32')
#         ytmp = np.array(list(map(lambda x: [index[a] for a in list(x)], l_plateStr)), dtype=np.uint8)
#         y = np.zeros([ytmp.shape[1], batch_size, len(chars)])
#         for batch in range(batch_size):
#             for idx, row_i in enumerate(ytmp[batch]):
#                 y[idx, batch, row_i] = 1
#         yield X, [yy for yy in y]

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.0001,momentum=0.9, decay=0.0, nesterov=False)

Mymodel = MyEndToEnd_Five.MagicModel((72, 272, 1))
Mymodel.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
Mymodel.load_weights("Models_weights/NewR_Five_plate_best01.h5")
#best_model = ModelCheckpoint("Models_weights/NewR_Five_plate_best02.h5", monitor='val_loss', verbose=1, save_best_only=True)
#Mymodel.fit(x=x_train, y=y_train, batch_size=64, epochs=3,verbose=1,shuffle=True,validation_data=(x_test,y_test),callbacks =[best_model])
#Mymodel.save_weights("NewR_Five_plate_best01.h5")
# Mymodel.fit_generator(gen_data(), steps_per_epoch=1000, epochs=5,verbose=1,validation_data=gen_data(), validation_steps=500,callbacks=[best_model])
o = Mymodel.evaluate(x_test, y_test,batch_size=64,verbose=1,)
print(o)
# pre = Mymodel.predict(x_test)
# pre = np.array(pre)
# #print(pre)
# print(np.max(pre,axis=2))
# print(pre.argmax(axis=2))
# #pre = np.array(pre.argmax(axis=2),dtype=np.int)
# #for a in pre:


