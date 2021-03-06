import Model.blockFactory as factory

def firstBlock(input):
    tensor = factory.zeroPadding(input, (3,3))
    tensor = factory.convolutionBlock(tensor, '', '1', 64, (7,7), (2,2))
    tensor = factory.zeroPadding(tensor)
    tensor = factory.maxPooling(tensor, (3,3), (2,2))

    return tensor

def secondBlock(input):
    tensor = factory.convolutionBlock(input, '', '2', 64, (1,1))
    tensor = factory.zeroPadding(tensor)
    #tensor = factory.maxPooling(tensor, (3,3), (2,2))
    #todo: do we need here maxpool?

    return tensor

def thirdBlock(input):
    tensor = factory.convolutionBlock(input, '', '3', 192, (3,3))
    tensor = factory.zeroPadding(tensor)
    tensor = factory.maxPooling(tensor, (3,3), (2,2))

    return tensor

def constructor(input):
    tensor = firstBlock(input)
    tensor = secondBlock(tensor)
    tensor = thirdBlock(tensor)

    #tensor = factory.merge([first, second, third])

    return tensor

    