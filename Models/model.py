# -*- coding:utf-8 -*-
import pydot
import keras
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense, Input, BatchNormalization,Activation
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from IPython.display import SVG

mnist = input_data.read_data_sets(".\\MNIST_data", one_hot=True)
x_train, y_train = mnist.train.images,mnist.train.labels
x_test, y_test = mnist.test.images, mnist.test.labels
x_train = x_train.reshape(-1, 28, 28,1).astype('float32')
x_test = x_test.reshape(-1,28, 28,1).astype('float32')

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

def MagicModel(input_shape = (72, 272, 3) ):
    input_tensor = Input(input_shape)
    x = input_tensor
    x = Conv2D(filters = 64, kernel_size = (3, 3), padding='same',kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3, 3), padding='same',kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(filters = 128, kernel_size = (3, 3), padding='same',kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = 128, kernel_size = (3, 3), padding='same',kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(filters = 256, kernel_size = (3, 3), padding='same',kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = 256, kernel_size = (3, 3), padding='same',kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    #n_class = len(chars)
    #x = [Dense(7, activation='softmax', name='c%d'%(i+1))(x) for i in range(7)]
    y = Dense(10, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=y)
    return model
magicmodel = MagicModel()
magicmodel.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])

# magicmodel.fit(x=x_train, y=y_train, batch_size=32, epochs=8)
#magicmodel.save_weights('my_model_weights1.h5')
#o = magicmodel.evaluate(x_test, y_test,batch_size=32,verbose=1,sample_weight='my_model_werghts1.h5')
#print(o)


plot_model(magicmodel, to_file='HappyModel05.png',show_shapes=True)
SVG(model_to_dot(model=magicmodel, show_layer_names=True, show_shapes=True).create(prog='dot', format='svg'))