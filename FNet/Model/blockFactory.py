from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense

dataFormat = 'channels_first'
activationFunc = 'relu'

def convolutionBlock(input, layerName, filters, kernelSize=(1,1), strides=(1,1), padding=None):
    tensor = Conv2D(filters, kernelSize, strides, padding, dataFormat, name=layerName)(input)
    tensor = BatchNormalization(axis=1, epsilon=0.00001)(tensor)
    tensor = Activation(activationFunc)(tensor)

    return tensor

def zeroPadding(input, padding=(1,1)):
    tensor = ZeroPadding2D(padding, dataFormat)(input)

    return tensor

def maxPooling(input, poolSize=(1,1), strides=(1,1), padding='valid'):
    tensor = MaxPooling2D(poolSize, strides, padding, dataFormat)(input)

    return tensor

def averagePooling(input, poolSize=(1,1), strides=(1,1), padding='valid'):
    tensor = AveragePooling2D(poolSize, strides, padding, dataFormat)(input)

    return tensor

def concatenate(input, axis=1):
    tensor = concatenate(input, axis)

    return tensor

def finishing(input, units, layerName):
    tensor = averagePooling(input, (3,3))
    tensor = Flatten(tensor)
    tensor = Dense(units, layerName)

    return tensor