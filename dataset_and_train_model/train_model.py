# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 18:19:35 2021

@author: samra
"""

import os
import shutil

import matplotlib.pyplot as plt

from keras.layers import *
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers


PATH = 'data/'

train_count = 0
test_count = 0
classes = os.listdir(os.path.join(PATH, "test/"))

for cl in classes:
    print(cl, ":", len(os.listdir(os.path.join(PATH, "train", cl))))
    train_count += len(os.listdir(os.path.join(PATH, "train", cl)))

print("total train images:", train_count)

for cl in classes:
    print(cl, ":", len(os.listdir(os.path.join(PATH, "test", cl))))
    test_count += len(os.listdir(os.path.join(PATH, "test", cl)))

print("total test images:", test_count)

# copying angry, surprised and disgusted pics from test folder to train folder

names = os.listdir("data/train/surprised")
for name in names:
    new = name.replace('im', 'img')
    os.rename(os.path.join("data/train/surprised", name), os.path.join("data/train/surprised", new))

images = os.listdir("data/test/surprised")
for i in images:
    shutil.copy(os.path.join("data/test/surprised", i), "data/train/surprised")
    if len(os.listdir("data/train/surprised")) == 4000:
        break

names = os.listdir("data/train/disgusted")
for name in names:
    new = name.replace('im', 'img')
    os.rename(os.path.join("data/train/disgusted", name), os.path.join("data/train/disgusted", new))

images = os.listdir("data/test/disgusted")
for i in images:
    shutil.copy(os.path.join("data/test/disgusted", i), "data/train/disgusted")

names = os.listdir("data/train/angry")
for name in names:
    new = name.replace('im', 'img')
    os.rename(os.path.join("data/train/angry", name), os.path.join("data/train/angry", new))

images = os.listdir("data/test/angry")
for i in images:
    shutil.copy(os.path.join("data/test/angry", i), "data/train/angry")
    if len(os.listdir("data/train/angry")) == 4000:
        break

# Deleting pics from all categories with more than 4000 pics(train)

del_list = []

for cl in classes:
    train_path = os.listdir(os.path.join("data/train", cl))
    if len(train_path) > 4000:
        for i in train_path[4000:]:
            del_list.append(os.path.join("data/train", cl, i))
print(len(del_list))

for i in del_list:
    os.remove(i)

# Deleting pics from all categories with more than 1000 pics(test)

del_list = []

for cl in classes:
    test_path = os.listdir(os.path.join("data/test", cl))
    if len(test_path) > 1000:
        for i in test_path[1000:]:
            del_list.append(os.path.join("data/test", cl, i))
print(len(del_list))

for i in del_list:
    os.remove(i)

PATH = 'data/'

train_count = 0
test_count = 0
classes = os.listdir(os.path.join(PATH, "test/"))

for cl in classes:
    print(cl, ":", len(os.listdir(os.path.join(PATH, "train", cl))))
    train_count += len(os.listdir(os.path.join(PATH, "train", cl)))

print("total train images:", train_count)

for cl in classes:
    print(cl, ":", len(os.listdir(os.path.join(PATH, "test", cl))))
    test_count += len(os.listdir(os.path.join(PATH, "test", cl)))

print("total test images:", test_count)

IMAGE_SIZE = [48, 48]
epochs = 150
BATCH_SIZE = 32
PATH = './data/'

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001),
                 input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(7, kernel_size=(1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(BatchNormalization())
model.add(Conv2D(7, kernel_size=(4, 4), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()

val_datagen = ImageDataGenerator(rescale=1. / 255)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    os.path.join(PATH, "train"),
    target_size=(48, 48),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
    os.path.join(PATH, "test"),
    target_size=(48, 48),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode='categorical')

r = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    steps_per_epoch=train_count // BATCH_SIZE,
    validation_steps=test_count // BATCH_SIZE
)

model.save("model.hdf5")

plt.plot(r.history['loss'])
plt.plot(r.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(r.history['accuracy'])
plt.plot(r.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
