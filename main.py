# -*- coding: utf-8 -*-
"""
Created on Thu May 27 15:42:30 2021

@author: samra
"""
import numpy as np
import cv2
import os

import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image

#Emotion class labels mapping the dataset folders
emotion = {
0: "angry",
1: "disgusted",
2: "fearful",
3: "happy",
4: "neutral",
5: "sad",
6: "surprised"
}

#Function that get image path and model, predict the emotion class and plot it. 
def emotionClassification(image,model):
    image_array = cv2.imread(image)
    gray_image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    gray_image_array = gray_image_array/255.0
    # recognizing emotion
    gray_image_array = np.array(gray_image_array)
    gray_image_array = np.resize(gray_image_array, (1, 48, 48, 1))
    pred = model.predict(gray_image_array)
    pred = np.argmax(pred)
    emotion_prediction = emotion[pred]
    plt.title(emotion_prediction)
    plt.imshow(image_array)
    plt.show()

#Loading saved model
model = tf.keras.models.load_model('saved_models/model.hdf5')
model.compile(metrics='accuracy')

#Predict an example of emotion
emotionClassification('happy.jpg',model)