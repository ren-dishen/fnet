from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense
from keras.models import Model
from keras import backend as Keras

dataFormat = 'channels_first'
activationFunc = 'relu'

def initialization(shape):
    initialTensor = Input(shape)

    return initialTensor

def convolutionBlock(input, layerName, blockNumber, filters, kernelSize=(1,1), strides=(1,1), padding='valid'):
    tensor = Conv2D(filters, kernelSize, strides=strides, padding=padding, data_format = dataFormat, name=layerName + 'conv' + blockNumber)(input)
    tensor = BatchNormalization(axis=1, epsilon=0.00001, name=layerName + 'bn' + blockNumber)(tensor)
    tensor = Activation(activationFunc)(tensor)

    return tensor

def zeroPadding(input, padding=(1,1)):
    tensor = ZeroPadding2D(padding, data_format = dataFormat)(input)

    return tensor

def maxPooling(input, poolSize=(1,1), strides=(1,1), padding='valid'):
    tensor = MaxPooling2D(poolSize, strides, padding, data_format = dataFormat)(input)

    return tensor

def averagePooling(input, poolSize=(1,1), strides=(1,1), padding='valid'):
    tensor = AveragePooling2D(poolSize, strides, padding, data_format = dataFormat)(input)

    return tensor

def merge(input, axis=1):
    tensor = concatenate(input, axis)

    return tensor

def finishing(input, output, units, layerName):
    tensor = averagePooling(output, (3,3))
    tensor = Flatten()(tensor)
    tensor = Dense(units, name=layerName)(tensor)

    tensor = Lambda(lambda  x: Keras.l2_normalize(x,axis=1))(tensor)
    model = Model(inputs = input, outputs = tensor, name='FNetModel')
        
    return model