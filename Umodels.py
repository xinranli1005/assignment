#
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras import models, layers
from keras.layers import Dense

def model_0(input_size):
    inputs = Input(input_size)
#encode
    model = models.Sequential()
    model.add(layers.Conv2D(64, (2, 2), strides = 1, padding = 'same', input_shape = input_size))
    model.add(layers.Conv2D(32, (2, 2), strides = 1, padding = 'same'))
    model.add(layers.Conv2D(16, (2, 2), strides = 1, padding = 'same'))
#
#latent
    model.add(layers.Conv2D(8, (2, 2), strides = 1, padding = 'same'))

#decode
    model.add(layers.Conv2DTranspose(16, (2, 2), strides = 1, padding = 'same'))
    model.add(layers.Conv2DTranspose(32, (2, 2), strides = 1, padding = 'same'))
    model.add(layers.Conv2DTranspose(64, (2, 2), strides = 1, padding = 'same'))
    model.add(layers.Conv2DTranspose(3, (3, 3), strides = 1, activation = 'sigmoid', padding = 'same'))
#
    return model
    
def model_1(input_size):
    inputs = Input(input_size)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same')(inputs)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv2)  
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#
    convm = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool2)
    convm = Conv2D(256, 3, activation = 'relu', padding = 'same')(convm)  
    dropconvm = Dropout(0.25)(convm)
#
    deconv2 = Conv2DTranspose(32, 3, strides=(2, 2), activation = 'relu', padding = 'same')(dropconvm)
    uconv2 = concatenate([deconv2,conv2], axis = 3) 
    uconv2 = Conv2D(32, 3, activation = 'relu', padding = 'same')(uconv2)
    uconv2 = Conv2D(32, 3, activation = 'relu', padding = 'same')(uconv2)
#
    deconv1 = Conv2DTranspose(16, 3, strides=(2, 2), activation = 'relu', padding = 'same')(uconv2)
    uconv1 = concatenate([deconv1,conv1], axis = 3) 
    uconv1 = Conv2D(16, 3, activation = 'relu', padding = 'same')(uconv1)
    uconv1 = Conv2D(16, 3, activation = 'relu', padding = 'same')(uconv1)
    output_layer = Conv2D(1, 1, activation = 'sigmoid', padding = 'same')(uconv1)
#
    model = Model(inputs = inputs, outputs = output_layer)
#
    return model


def model_1s(input_size):
    inputs = Input(input_size)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same')(inputs)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # conv1 is 28
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv2)  
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#
    convm = Conv2D(64, 3, activation = 'relu', padding = 'same')(pool2)
    convm = Conv2D(64, 3, activation = 'relu', padding = 'same')(convm)  
    dropconvm = Dropout(0.25)(convm)
#
    deconv2 = Conv2DTranspose(32, 3, strides=(2, 2), activation = 'relu', padding = 'same')(dropconvm)
    uconv2 = concatenate([deconv2,conv2], axis = 3) 
    uconv2 = Conv2D(32, 3, activation = 'relu', padding = 'same')(uconv2)
    uconv2 = Conv2D(32, 3, activation = 'relu', padding = 'same')(uconv2)
#
    deconv1 = Conv2DTranspose(16, 3, strides=(2, 2), activation = 'relu', padding = 'same')(uconv2)
    uconv1 = concatenate([deconv1,conv1], axis = 3) 
    uconv1 = Conv2D(16, 3, activation = 'relu', padding = 'same')(uconv1)
    uconv1 = Conv2D(16, 3, activation = 'relu', padding = 'same')(uconv1)
    output_layer = Conv2D(1, 1, activation = 'sigmoid', padding = 'same')(uconv1)
#
    model = Model(inputs = inputs, outputs = output_layer)
#
    return model



def model_2(input_size):
    inputs = Input(input_size)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same')(inputs)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#
    convm = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool2)
    convm = Conv2D(256, 3, activation = 'relu', padding = 'same')(convm)
    dropconvm = Dropout(0.25)(convm)
#
    deconv2 = Conv2DTranspose(32, 3, strides=(2, 2), activation = 'relu', padding = 'same')(dropconvm)
    uconv2 = concatenate([deconv2,conv2], axis = 3) 
    uconv2 = Conv2DTranspose(32, (2, 2), strides = 1, padding = 'same')(uconv2)
    uconv2 = Conv2DTranspose(32, (2, 2), strides = 1, padding = 'same')(uconv2)
#
    deconv1 = Conv2DTranspose(16, 3, strides=(2, 2), activation = 'relu', padding = 'same')(uconv2)
    uconv1 = concatenate([deconv1,conv1], axis = 3) 
    uconv1 = Conv2DTranspose(16, (2, 2), strides = 1, padding = 'same')(uconv1)
    uconv1 = Conv2DTranspose(16, (2, 2), strides = 1, padding = 'same')(uconv1)
    output_layer = Conv2DTranspose(1, (1,1), strides = 1,activation = 'sigmoid', padding = 'same')(uconv1)
#
    model = Model(inputs = inputs, outputs = output_layer)
#
    return model


def model_2s(input_size):
    inputs = Input(input_size)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same')(inputs)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv2)  
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#
    convm = Conv2D(64, 3, activation = 'relu', padding = 'same')(pool2)
    convm = Conv2D(64, 3, activation = 'relu', padding = 'same')(convm)  
    dropconvm = Dropout(0.25)(convm)
#
    deconv2 = Conv2DTranspose(32, 3, strides=(2, 2), activation = 'relu', padding = 'same')(dropconvm)
    uconv2 = concatenate([deconv2,conv2], axis = 3) 
    uconv2 = Conv2DTranspose(32, (2, 2), strides = 1, padding = 'same')(uconv2)
    uconv2 = Conv2DTranspose(32, (2, 2), strides = 1, padding = 'same')(uconv2)
#
    deconv1 = Conv2DTranspose(16, 3, strides=(2, 2), activation = 'relu', padding = 'same')(uconv2)
    uconv1 = concatenate([deconv1,conv1], axis = 3) 
    uconv1 = Conv2DTranspose(16, (2, 2), strides = 1, padding = 'same')(uconv1)
    uconv1 = Conv2DTranspose(16, (2, 2), strides = 1, padding = 'same')(uconv1)
    output_layer = Conv2DTranspose(1, (1,1), strides = 1,activation = 'sigmoid', padding = 'same')(uconv1)
#
    model = Model(inputs = inputs, outputs = output_layer)
#
    return model


def model_3(input_size):
    inputs = Input(input_size)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2) 
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#
    convm = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    convm = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(convm)  
    dropconvm = Dropout(0.25)(convm)
#
    deconv2 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(dropconvm))
    uconv2 = concatenate([deconv2,conv2], axis = 3)  
    uconv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(uconv2)
    uconv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(uconv2)
#
    deconv1 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(uconv2))
    uconv1 = concatenate([deconv1,conv1], axis = 3)  
    uconv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(uconv1)
    uconv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(uconv1)
    output_layer = Conv2D(1, 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(uconv1)
#
    model = Model(inputs = inputs, outputs = output_layer)
#
    return model


def model_3s(input_size):
    inputs = Input(input_size)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2) 
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#
    convm = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    convm = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(convm)  
    dropconvm = Dropout(0.25)(convm)
#
    deconv2 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(dropconvm))
    uconv2 = concatenate([deconv2,conv2], axis = 3)  
    uconv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(uconv2)
    uconv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(uconv2)
#
    deconv1 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(uconv2))
    uconv1 = concatenate([deconv1,conv1], axis = 3)  
    uconv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(uconv1)
    uconv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(uconv1)
    output_layer = Conv2D(1, 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(uconv1)
#
    model = Model(inputs = inputs, outputs = output_layer)
#
    return model


def model_4(input_size):
    inputs = Input(input_size)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same')(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv1)
    
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv2)  
#
    conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv2)
    conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv3)
    dropconvm = Dropout(0.25)(conv3)
#
    conv4 = Conv2D(32, 3, activation = 'relu', padding = 'same')(dropconvm)
    conv4 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv4)
#
    conv5 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv4)
    conv5 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv5)
    output_layer = Conv2D(1, 1, activation = 'sigmoid', padding = 'same')(conv5)
#
    model = Model(inputs = inputs, outputs = output_layer)
#
    return model


def model_4s(input_size):
    inputs = Input(input_size)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same')(inputs)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same')(conv1)
    
    conv2 = Conv2D(16, 3, activation = 'relu', padding = 'same')(conv1)
    conv2 = Conv2D(16, 3, activation = 'relu', padding = 'same')(conv2)  
#
    conv3 = Conv2D(16, 3, activation = 'relu', padding = 'same')(conv2)
    conv3 = Conv2D(16, 3, activation = 'relu', padding = 'same')(conv3)
    dropconvm = Dropout(0.25)(conv3)
#
    conv4 = Conv2D(16, 3, activation = 'relu', padding = 'same')(dropconvm)
    conv4 = Conv2D(16, 3, activation = 'relu', padding = 'same')(conv4)
#
    conv5 = Conv2D(16, 3, activation = 'relu', padding = 'same')(conv4)
    conv5 = Conv2D(16, 3, activation = 'relu', padding = 'same')(conv5)
    output_layer = Conv2D(1, 1, activation = 'sigmoid', padding = 'same')(conv5)
#
    model = Model(inputs = inputs, outputs = output_layer)
#
    return model
    

def model_5(input_size):
    inputs = Input(input_size)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same')(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv1)
    
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv2)  
    rconv2 = concatenate([conv1,conv2], axis = 3) 
#
    conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same')(rconv2)
    conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv3)
    rconv3 = concatenate([rconv2,conv3], axis = 3) 
    dropconvm = Dropout(0.25)(rconv3)
#
    conv4 = Conv2D(32, 3, activation = 'relu', padding = 'same')(dropconvm)
    conv4 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv4)
    rconv4 = concatenate([rconv3,conv4], axis = 3) 
#
    conv5 = Conv2D(32, 3, activation = 'relu', padding = 'same')(rconv4)
    conv5 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv5)
    output_layer = Conv2D(3, 3, activation = 'sigmoid', padding = 'same')(conv5)
#
    model = Model(inputs = inputs, outputs = output_layer)
#
    return model


def model_5s(input_size):
    inputs = Input(input_size)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same')(inputs)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same')(conv1)
    
    conv2 = Conv2D(16, 3, activation = 'relu', padding = 'same')(conv1)
    conv2 = Conv2D(16, 3, activation = 'relu', padding = 'same')(conv2)  
    rconv2 = concatenate([conv1,conv2], axis = 3) 
#
    conv3 = Conv2D(16, 3, activation = 'relu', padding = 'same')(rconv2)
    conv3 = Conv2D(16, 3, activation = 'relu', padding = 'same')(conv3)
    rconv3 = concatenate([rconv2,conv3], axis = 3) 
    dropconvm = Dropout(0.25)(rconv3)
#
    conv4 = Conv2D(16, 3, activation = 'relu', padding = 'same')(dropconvm)
    conv4 = Conv2D(16, 3, activation = 'relu', padding = 'same')(conv4)
    rconv4 = concatenate([rconv3,conv4], axis = 3) 
#
    conv5 = Conv2D(16, 3, activation = 'relu', padding = 'same')(rconv4)
    conv5 = Conv2D(16, 3, activation = 'relu', padding = 'same')(conv5)
    output_layer = Conv2D(1, 1, activation = 'sigmoid', padding = 'same')(conv5)
#
    model = Model(inputs = inputs, outputs = output_layer)
#
    return model



def model_6(input_size):
    inputs = Input(input_size)
    uconv1 = Conv2DTranspose(16, 3, strides=(2, 2), activation = 'relu', padding = 'same')(inputs)
    uconv2 = Conv2DTranspose(32, (2, 2), strides = 1, padding = 'same')(uconv1)
    uconv2 = Conv2DTranspose(32, (2, 2), strides = 1, padding = 'same')(uconv2)

    pool1 = MaxPooling2D(pool_size=(2, 2))(uconv2)  # 

    conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same')(pool1)
    conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv3) 

    dropconv3 = Dropout(0.25)(conv3)

    conv4 = Conv2D(32, 3, activation = 'relu', padding = 'same')(dropconv3)
    conv4 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv4) 

    output_layer = Conv2D(1, 1, activation = 'sigmoid', padding = 'same')(conv4)
#
    model = Model(inputs = inputs, outputs = output_layer)
#
    return model

