import Model.blockFactory as factory

def block1x1(input):
    tensor = factory.convolutionBlock(input, 'inception_4a_1x1_', '', 256, (1,1))

    return tensor

def block3x3(input):
    tensor = factory.convolutionBlock(input, 'inception_4a_3x3_', '1', 96, (1,1))
    tensor = factory.zeroPadding(tensor)
    tensor = factory.convolutionBlock(tensor, 'inception_4a_3x3_', '2', 192, (3,3))

    return tensor

def block5x5(input):
    tensor = factory.convolutionBlock(input, 'inception_4a_5x5_', '1', 32, (1,1))
    tensor = factory.zeroPadding(tensor, (2,2))
    tensor = factory.convolutionBlock(tensor, 'inception_4a_5x5_', '2', 64, (5,5))

    return tensor

def blockPool(input):
    tensor = factory.averagePooling(input, (3,3), (3,3))
    tensor = factory.convolutionBlock(tensor, 'inception_4a_pool_', '', 128, (1,1))
    tensor = factory.zeroPadding(tensor, (2,2))

    return tensor

def inceptionConstructor(input):
    _3x3 = block3x3(input)
    _5x5 = block5x5(input)
    _pool = blockPool(input)
    _1x1 = block1x1(input)

    tensor = factory.merge([_3x3, _5x5, _pool, _1x1])

    return tensor

    