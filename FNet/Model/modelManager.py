import tensorflow
import numpy as np
import os
from numpy import genfromtxt
import Utilities
import Model.Constants as Constants
import Model.blockFactory as factory
import Model.Layers.Start.firstBlock as startBlock
import Model.Layers.Inceptions.FirstInception.firstBlock as firstBlockOfFirstInception
import Model.Layers.Inceptions.FirstInception.secondBlock as secondBlockOfFirstInception
import Model.Layers.Inceptions.FirstInception.thirdBlock as thirdBlockOfFirstInception
import Model.Layers.Inceptions.SecondInception.firstBlock as firstBlockOfSecondInception
import Model.Layers.Inceptions.SecondInception.secondBlock as secondBlockOfSecondInception
import Model.Layers.Inceptions.ThirdInception.firstBlock as firstBlockOfThirdInception
import Model.Layers.Inceptions.ThirdInception.secondBlock as secondBlockOfThirdInception

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

    model = factory.finishing(initialTensor, tensor, 128, 'dense_layer')

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

def verify(image_path, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".
    
    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras
    
    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """
    
    ### START CODE HERE ###
    
    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
    encoding = Utilities.GetImageData(image_path, model)
    
    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm((encoding - database[identity]))
    
    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if dist < 0.7:
        print("It's " + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False
        
    ### END CODE HERE ###
        
    return dist, door_open

def loadWeightsFromFile():
    # Set weights path
    dirPath = './Model/weights'
    fileNames = filter(lambda x: not x.startswith('.'), os.listdir(dirPath))
    paths = {}
    weights_dict = {}

    for fileName in fileNames:
        paths[fileName.replace('.csv', '')] = dirPath + '/' + fileName

    for name in Constants.layerNames:
        if 'conv' in name:
            conv_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
            conv_w = np.reshape(conv_w, Constants.conv_shape[name])
            conv_w = np.transpose(conv_w, (2, 3, 1, 0))
            conv_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
            weights_dict[name] = [conv_w, conv_b]     
        elif 'bn' in name:
            bn_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
            bn_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
            bn_m = genfromtxt(paths[name + '_m'], delimiter=',', dtype=None)
            bn_v = genfromtxt(paths[name + '_v'], delimiter=',', dtype=None)
            weights_dict[name] = [bn_w, bn_b, bn_m, bn_v]
        elif 'dense' in name:
            dense_w = genfromtxt(dirPath+'/dense_w.csv', delimiter=',', dtype=None)
            dense_w = np.reshape(dense_w, (128, 736))
            dense_w = np.transpose(dense_w, (1, 0))
            dense_b = genfromtxt(dirPath+'/dense_b.csv', delimiter=',', dtype=None)
            weights_dict[name] = [dense_w, dense_b]

    return weights_dict

def loadWeights(model):
    # Load weights from csv files (which was exported from Openface torch model)
    weights_dict = loadWeightsFromFile()

    # Set layer weights of the model
    for name in Constants.layerNames:
        if model.get_layer(name) != None:
            model.get_layer(name).set_weights(weights_dict[name])
        #elif model.get_layer(name) != None:
        #    model.get_layer(name).set_weights(weights_dict[name])

