import Model.blockFactory as factory

def block1x1(input):
    tensor = factory.convolutionBlock(input, 'inception_3a_1x1_', '', 64, (1,1))

    return tensor

def block3x3(input):
    tensor = factory.convolutionBlock(input, 'inception_3a_3x3_', '1', 96, (1,1))
    tensor = factory.zeroPadding(tensor)
    tensor = factory.convolutionBlock(tensor, 'inception_3a_3x3_', '2', 128, (3,3))

    return tensor

def block5x5(input):
    tensor = factory.convolutionBlock(input, 'inception_3a_5x5_', '1', 16, (1,1))
    tensor = factory.zeroPadding(tensor, (2,2))
    tensor = factory.convolutionBlock(tensor, 'inception_3a_5x5_', '2', 32, (5,5))

    return tensor

def blockPool(input):
    tensor = factory.maxPooling(input, (3,3), (2,2))
    tensor = factory.convolutionBlock(tensor, 'inception_3a_pool_', '', 32, (1,1))
    tensor = factory.zeroPadding(tensor, ((3,4),(3,4)))

    return tensor

def inceptionConstructor(input):
    _3x3 = block3x3(input)
    _5x5 = block5x5(input)
    _pool = blockPool(input)
    _1x1 = block1x1(input)

    tensor = factory.merge([_3x3, _5x5, _pool, _1x1])

    return tensor

    