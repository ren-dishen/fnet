from FNet.Model import blockFactory as factory

def block3x3(input):
    tensor = factory.convolutionBlock(input, 'IBlock/2/2/3x3/1', 160, (1,1))
    tensor = factory.zeroPadding(tensor)
    tensor = factory.convolutionBlock(tensor, 'IBlock/2/2/3x3/2', 256, (3,3), (2,2))

    return tensor

def block5x5(input):
    tensor = factory.convolutionBlock(input, 'IBlock/2/2/5x5/1', 64, (1,1))
    tensor = factory.zeroPadding(tensor, (2,2))
    tensor = factory.convolutionBlock(tensor, 'IBlock/2/2/5x5/2', 128, (5,5), (2,2))

    return tensor

def blockPool(input):
    tensor = factory.averagePooling(input, (3,3), (2,2))
    tensor = factory.zeroPadding(tensor, (0,1), (0,1))

    return tensor

def inceptionConstructor(input):
    _3x3 = block3x3(input)
    _5x5 = block5x5(input)
    _pool = blockPool(input)

    tensor = factory.merge([_3x3, _5x5, _pool])

    return tensor

    