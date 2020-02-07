import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from scipy.io import wavfile
import os
import Globals
import math

CHECKPOINT_FILE_NAME = 'model.ckpt'

SET_INPUT = 0
SET_TARGET = 1


class Model:

    def __init__(self, version):
        self.version = version
        self.checkpoint_file_path = os.path.join(Globals.MODEL_PATH, version + '.' + CHECKPOINT_FILE_NAME)
        self.model = None

        self.__create_model(version)

    def __create_model(self, version):
        self.__create_model_by_version(version)

        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_file_path,
            save_weights_only=True,
            verbose=1)

    def load(self):
        self.model.load_weights(self.checkpoint_file_path)

    def fit(self, training_set, validation_set, epochs=20):

        processed_training_set = self.__process_sets(training_set)
        processed_validation_set = self.__process_sets(validation_set)

        self.model.fit(processed_training_set[SET_INPUT],
                       processed_training_set[SET_TARGET],
                       batch_size=64,
                       epochs=epochs,
                       validation_data=(processed_validation_set[SET_INPUT], processed_validation_set[SET_TARGET]),
                       callbacks=[self.cp_callback])

    def predict(self, input):
        input_values = input.reshape((-1, 128, 513, 1))
        return self.model.predict(input_values)

    def evaluate(self, test_set):
        self.__create_model_by_version(self.version)
        self.model.load_weights(self.checkpoint_file_path)

        processed_test_set = self.__process_sets(test_set)

        return self.model.evaluate(processed_test_set[SET_INPUT], processed_test_set[SET_TARGET], verbose=2)

    def __process_sets(self, set):
        target_values = []
        input_values = []

        for set_tuple in set:
            input_values.append(set_tuple[0])
            target_values.append(set_tuple[1])

        input_values = np.array(input_values)
        input_values = input_values.reshape((-1, 128, 513, 1))

        target_values = np.array(target_values)

        return [input_values, target_values]

    def __create_model_by_version(self, version):
        if version == Globals.MODEL_VERSION_1:
            self.model = self.__create_model_version1()
        else:
            self.model = self.__create_model_version2()

    def __create_model_version1(self):
        model = models.Sequential()

        model.add(layers.InputLayer(input_shape=(128, 513, 1)))
        model.add(layers.Conv2D(256, (4, 513), activation='relu'))
        model.add(layers.MaxPooling2D((2, 1)))
        model.add(layers.Conv2D(128, (4, 1), activation='relu'))
        model.add(layers.MaxPooling2D((2, 1)))
        model.add(layers.Conv2D(256, (4, 1), activation='relu'))
        model.add(layers.AveragePooling2D((26, 1)))

        model.add(layers.Flatten())
        model.add(layers.Dense(300, activation='relu'))
        model.add(layers.Dense(150, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        model.summary()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def __create_model_version2(self):
        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=(128, 513, 1)))
        model.add(layers.Conv2D(256, (4, 513), activation='relu'))
        model.add(layers.Conv2D(256, (4, 1), activation='relu'))
        model.add(layers.MaxPooling2D((2, 1)))
        model.add(layers.Conv2D(128, (4, 1), activation='relu'))
        model.add(layers.Conv2D(256, (4, 1), activation='relu'))
        # model.add(layers.MaxPooling2D((2, 1)))
        model.add(layers.AveragePooling2D((55, 1)))

        model.add(layers.Flatten())
        # model.add(layers.Dense(300, activation='relu'))
        model.add(layers.Dense(150, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        model.summary()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model
