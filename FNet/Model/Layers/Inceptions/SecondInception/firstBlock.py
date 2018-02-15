from FNet.Model import blockFactory as factory

def block1x1(input):
    tensor = factory.convolutionBlock(input, 'IBlock/2/1/1x1/1', 256, (1,1))

    return tensor

def block3x3(input):
    tensor = factory.convolutionBlock(input, 'IBlock/2/1/3x3/1', 96, (1,1))
    tensor = factory.zeroPadding(tensor)
    tensor = factory.convolutionBlock(tensor, 'IBlock/2/1/3x3/2', 192, (3,3))

    return tensor

def block5x5(input):
    tensor = factory.convolutionBlock(input, 'IBlock/2/1/5x5/1', 32, (1,1))
    tensor = factory.zeroPadding(tensor, (2,2))
    tensor = factory.convolutionBlock(tensor, 'IBlock/2/1/5x5/2', 64, (5,5))

    return tensor

def blockPool(input):
    tensor = factory.averagePooling(input, (3,3), (3,3))
    tensor = factory.convolutionBlock(tensor, 'IBlock/2/1/pool/1', 128, (1,1))
    tensor = factory.zeroPadding(tensor, (2,2))

    return tensor

def inceptionConstructor(input):
    _3x3 = block3x3(input)
    _5x5 = block5x5(input)
    _pool = blockPool(input)
    _1x1 = block1x1(input)

    tensor = factory.concatenate([_3x3, _5x5, _pool, _1x1])

    return tensor

    