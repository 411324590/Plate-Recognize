# -*- coding:utf-8 -*-
import pydot
import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense, Input, BatchNormalization,Activation,GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from IPython.display import SVG


adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

def MagicModel(input_shape = (72, 272, 1) ):
    input_tensor = Input(input_shape)
    x = input_tensor
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same',kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = 64, kernel_size=(3, 3), padding='same',kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(filters = 128, kernel_size=(3, 3), padding='same',kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = 128, kernel_size=(3, 3), padding='same',kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(filters = 256, kernel_size=(3, 3), padding='same',kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = 256, kernel_size=(3, 3), padding='same',kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(filters = 512, kernel_size=(3, 3), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = 512, kernel_size=(3, 3), padding='same',kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    #x = GlobalAveragePooling2D(data_format="channels_last")(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.35)(x)
    #x = Dense(1024,activation="relu")(x)
    #n_class = len(chars)
    x = [Dense(65, activation='softmax', name='c%d'%(i+1))(Dense(512, activation='softmax', name='c%d'%(i+11))(x)) for i in range(7)]
    # y = Dense(10, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=x)
    return model

# magicmodel = MagicModel((72, 272, 1))
# magicmodel.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
# magicmodel.summary()
# plot_model(magicmodel, to_file='ETEModel05_Four.png',show_shapes=True)
# SVG(model_to_dot(model=magicmodel, show_layer_names=True, show_shapes=True).create(prog='dot', format='svg'))