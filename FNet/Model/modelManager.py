import Model.blockFactory as factory
from keras.layers import Input
import Model.Layers.Start.firstBlock as startBlock
#import inc

def CreateModel(shape):
    initialTensor = Input(shape)
    startBlock.constructor(initialTensor)
    return model

CreateModel((3,96,96))