import Model.blockFactory as factory

def block3x3(input):
    tensor = factory.convolutionBlock(input, 'inception_4e_3x3_', '1', 160, (1,1))
    tensor = factory.zeroPadding(tensor)
    tensor = factory.convolutionBlock(tensor, 'inception_4e_3x3_','2', 256, (3,3), (2,2))

    return tensor

def block5x5(input):
    tensor = factory.convolutionBlock(input, 'inception_4e_5x5_', '1', 64, (1,1))
    tensor = factory.zeroPadding(tensor, (2,2))
    tensor = factory.convolutionBlock(tensor, 'inception_4e_5x5_', '2', 128, (5,5), (2,2))

    return tensor

def blockPool(input):
    tensor = factory.averagePooling(input, (3,3), (2,2))
    tensor = factory.zeroPadding(tensor, ((0,1), (0,1)))

    return tensor

def inceptionConstructor(input):
    _3x3 = block3x3(input)
    _5x5 = block5x5(input)
    _pool = blockPool(input)

    tensor = factory.merge([_3x3, _5x5, _pool])

    return tensor

    