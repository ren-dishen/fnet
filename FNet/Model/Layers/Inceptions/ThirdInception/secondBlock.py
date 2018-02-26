import Model.blockFactory as factory

def block1x1(input):
    tensor = factory.convolutionBlock(input, 'inception_5b_1x1_', '', 256, (1,1))

    return tensor

def block3x3(input):
    tensor = factory.convolutionBlock(input, 'inception_5b_3x3_', '1', 96, (1,1))
    tensor = factory.zeroPadding(tensor)
    tensor = factory.convolutionBlock(tensor, 'inception_5b_3x3_', '2', 384, (3,3))

    return tensor

def blockPool(input):
    tensor = factory.maxPooling(input, (3,3), (2,2))
    tensor = factory.convolutionBlock(tensor, 'inception_5b_pool_', '', 96, (1,1))
    tensor = factory.zeroPadding(tensor, (1,1))

    return tensor

def inceptionConstructor(input):
    _3x3 = block3x3(input)
    _pool = blockPool(input)
    _1x1 = block1x1(input)

    tensor = factory.merge([_3x3, _pool, _1x1])

    return tensor

    