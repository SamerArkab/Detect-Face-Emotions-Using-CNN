# -*- coding: utf-8 -*-
"""
Created on Thu May 27 15:42:30 2021

@author: samra
"""
import numpy as np
import cv2

import matplotlib.pyplot as plt
import tensorflow as tf

# Emotion class labels mapping the dataset folders
emotion = {
    0: "angry",
    1: "disgusted",
    2: "fearful",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprised"
}


# Function that get image path and model, predict the emotion class and plot it
def emotion_classification(image_path, run_model):
    image_array = cv2.imread(image_path)
    gray_image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    gray_image_array = gray_image_array / 255.0
    # recognizing emotion
    gray_image_array = np.array(gray_image_array)
    gray_image_array = np.resize(gray_image_array, (1, 48, 48, 1))
    predict = run_model.predict(gray_image_array)
    predict = np.argmax(predict)
    emotion_prediction = emotion[predict]
    plt.title(emotion_prediction)
    plt.imshow(image_array)
    plt.show()


# Loading saved (pre-trained) model
model = tf.keras.models.load_model('saved_models/model.hdf5')
model.compile(metrics='accuracy')

# Predict an example of emotion
emotion_classification('examples/face.jpg', model)
