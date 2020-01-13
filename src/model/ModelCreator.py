import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
import os
import math

from Helper import getListOfPreprocessedSongs, removeFiles, loadDataFromNumpyFile

preprocessedSongs = getListOfPreprocessedSongs()

# for file in preprocessedSongs:
#     processedSong = np.array(loadDataFromNumpyFile(file))


model = models.Sequential()

model.add(layers.InputLayer(input_shape=(128, 513, 1)))
model.add(layers.Conv2D(128, (4, 513), activation='relu'))
model.add(layers.MaxPooling2D((2, 1)))
model.add(layers.Conv2D(128, (4, 1), activation='relu'))
model.add(layers.MaxPooling2D((2, 1)))
model.add(layers.Conv2D(256, (4, 1), activation='relu'))
model.add(layers.MaxPooling2D((26, 1)))
model.add(layers.Dense(300, activation='relu'))
model.add(layers.Dense(150, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# history = model.fit(train_images, train_labels, epochs=10,
#                     validation_data=(test_images, test_labels))
