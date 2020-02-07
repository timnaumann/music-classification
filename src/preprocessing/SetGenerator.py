import numpy as np
import os

import Globals

from Helper import load_data_from_numpy_file, save_to_numpy_file, get_genres_from_data
from model.TargetValue import TARGET_VALUE_MAP

# 90% test set, 10% test set
TRAINING_TEST_SET_RATIO = 0.9


class SetGenerator:
    def __init__(self):
        self.test_set = []
        self.training_set = []

    def get_test_set(self):
        return load_data_from_numpy_file(os.path.join(Globals.BASE_PATH_TO_SOUND_FILES, Globals.TEST_SET_NAME))[0]

    def get_training_set(self):
        return load_data_from_numpy_file(os.path.join(Globals.BASE_PATH_TO_SOUND_FILES, Globals.TRAINING_SET_NAME))[0]

    def generate_training_and_test_set(self):
        genres = get_genres_from_data()

        for genre in genres:
            training_files = []
            test_files = []

            directory_path = os.path.join(Globals.BASE_PATH_TO_SOUND_FILES, genre, Globals.PROCESSED_FOLDER_NAME)

            if not os.path.exists(directory_path):
                continue

            print('Processing ' + genre + ' ...')
            processed_sound_files = os.listdir(directory_path)

            # maps file name pop.wav.npz to a relative path like ./data/genres/pop
            processed_sound_files = [os.path.join(directory_path, file) for file in processed_sound_files]

            np.random.shuffle(processed_sound_files)

            # 90% go into training set, 10% into validation set, 10% into test set
            split_index = round(len(processed_sound_files) * TRAINING_TEST_SET_RATIO)
            training_files.extend(processed_sound_files[:split_index])
            test_files.extend(processed_sound_files[split_index:])

            target_value = next(v.get_target_value() for v in TARGET_VALUE_MAP if v.get_name() == genre)

            self.training_set.extend(self.__generate_prediction_sets_based_on_file_list(training_files, target_value))
            self.test_set.extend(self.__generate_prediction_sets_based_on_file_list(test_files, target_value))

            print('Finished ' + genre)

        print('final length:' + str(len(self.training_set)))
        self.__create_training_file()
        self.__create_test_file()

    def __create_training_file(self):
        np.random.shuffle(self.training_set)
        save_to_numpy_file(os.path.join(Globals.BASE_PATH_TO_SOUND_FILES, Globals.TRAINING_SET_NAME), self.training_set)

    def __create_test_file(self):
        np.random.shuffle(self.test_set)
        save_to_numpy_file(os.path.join(Globals.BASE_PATH_TO_SOUND_FILES, Globals.TEST_SET_NAME), self.test_set)

    def __generate_prediction_sets_based_on_file_list(self, files, genre):
        resulting_file = []
        for file in files:
            resulting_file.extend(self._extract_values_from_file(file, genre))
        return resulting_file

    # output is a list of the shape [( 128 x 513 tensor, targetValue), ...]
    def _extract_values_from_file(self, file_path, target_prediction):
        processed_song = load_data_from_numpy_file(file_path)[0]

        set_of_prediction_values = []

        frames_needed_for_input = 128
        amount_of_frames = len(processed_song)
        index = 0

        while index <= amount_of_frames:
            frames_for_training_element = processed_song[index:index + frames_needed_for_input]
            predicting_value = tuple([frames_for_training_element, target_prediction])

            # add the training element only it has 128 frames inside
            if not len(frames_for_training_element) < frames_needed_for_input:
                set_of_prediction_values.append(predicting_value)

            index += int(frames_needed_for_input)

        return set_of_prediction_values
