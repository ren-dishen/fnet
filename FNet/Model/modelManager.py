import tensorflow
import Model.blockFactory as factory
import Model.Layers.Start.firstBlock as startBlock
import Model.Layers.Inceptions.FirstInception.firstBlock as firstBlockOfFirstInception
import Model.Layers.Inceptions.FirstInception.secondBlock as secondBlockOfFirstInception
import Model.Layers.Inceptions.FirstInception.thirdBlock as thirdBlockOfFirstInception
import Model.Layers.Inceptions.SecondInception.firstBlock as firstBlockOfSecondInception
import Model.Layers.Inceptions.SecondInception.secondBlock as secondBlockOfSecondInception
import Model.Layers.Inceptions.ThirdInception.firstBlock as firstBlockOfThirdInception
import Model.Layers.Inceptions.ThirdInception.secondBlock as secondBlockOfThirdInception
#import inc

def CreateModel(shape):
    initialTensor = factory.initialization(shape)
    tensor = startBlock.constructor(initialTensor)

    tensor = firstBlockOfFirstInception.inceptionConstructor(tensor)
    tensor = secondBlockOfFirstInception.inceptionConstructor(tensor)
    tensor = thirdBlockOfFirstInception.inceptionConstructor(tensor)

    tensor = firstBlockOfSecondInception.inceptionConstructor(tensor)
    tensor = secondBlockOfSecondInception.inceptionConstructor(tensor)

    tensor = firstBlockOfThirdInception.inceptionConstructor(tensor)
    tensor = secondBlockOfThirdInception.inceptionConstructor(tensor)

    model = factory.finishing(initialTensor, tensor, 128, 'denseBlock')

    return model

def CalculateTripletLoss(target, outputs, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = outputs[0], outputs[1], outputs[2]
    
    ### START CODE HERE ### (? 4 lines)
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    positiveDistance =  tensorflow.reduce_sum(tensorflow.square(tensorflow.subtract(anchor, positive)), axis = -1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    negativeDistance = tensorflow.reduce_sum(tensorflow.square(tensorflow.subtract(anchor, negative)), axis = -1)
    # Step 3: subtract the two previous distances and add alpha.
    basicLoss = positiveDistance - negativeDistance + alpha# tf.add(tf.subtract(pos_dist, neg_dist), alpha)#
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tensorflow.reduce_sum(tensorflow.maximum(basicLoss, 0.0))
    ### END CODE HERE ###
    
    return loss

