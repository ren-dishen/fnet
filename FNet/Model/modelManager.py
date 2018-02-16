from keras.layers import Input
import Model.blockFactory as factory
import Model.Layers.Start.firstBlock as startBlock
import Model.Layers.Inceptions.FirstInception as firstInception
#import inc

def CreateModel(shape):
    initialTensor = Input(shape)
    tensor = startBlock.constructor(initialTensor)
    tensor = firstInception.firstBlock.constructor(tensor)
    return model

CreateModel((3,96,96))