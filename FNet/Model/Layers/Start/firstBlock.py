import Model.blockFactory as factory

def firstBlock(input):
    tensor = factory.zeroPadding(input, (3,3))
    tensor = factory.convolutionBlock(tensor, 'IBlock/start/1/1/1', 64, (7,7), (2,2))
    tensor = factory.zeroPadding(tensor)
    tensor = factory.maxPooling(tensor, (3,3), (2,2))

    return tensor

def secondBlock(input):
    tensor = factory.convolutionBlock(tensor, 'IBlock/start/1/2/1', 64, (1,1))
    tensor = factory.zeroPadding(tensor)
    tensor = factory.maxPooling(tensor, (3,3), (2,2))
    #todo: do we need here maxpool?

    return tensor

def thirdBlock(input):
    tensor = factory.convolutionBlock(tensor, 'IBlock/start/1/3/1', 192, (3,3))
    tensor = factory.zeroPadding(tensor)
    tensor = factory.maxPooling(tensor, (3,3), (2,2))

    return tensor

def constructor(input):
    first = firstBlock(input)
    second = secondBlock(input)
    third = thirdBlock(input)

    tensor = factory.concatenate([_3x3, _5x5, _pool, _1x1])

    return tensor

    