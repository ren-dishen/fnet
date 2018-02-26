import cv2
import numpy as np

def GetImageData(path, model):
    print(path)
    tempImage = cv2.imread(path, 1)
    img = tempImage[...,::-1]

    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    data = np.array([img])

    embedding = model.predict_on_batch(data)
    return embedding