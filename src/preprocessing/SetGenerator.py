import numpy as np
import os
from Helper import load_data_from_numpy_file, save_to_numpy_file, get_genres_from_data
from model.TargetValue import TARGET_VALUE_MAP

basePathToSoundFiles = "./data/genres"

# 90% test set, 10% test set
trainingTestSetRatio = 0.9


class SetGenerator:
    def __init__(self):
        self.test_set = []
        self.training_set = []

    def get_test_set(self):
        return [name for name in os.listdir(basePathToSoundFiles) if
                os.path.isdir(os.path.join(basePathToSoundFiles, name))]

    def get_training_set(self):
        return [name for name in os.listdir(basePathToSoundFiles) if
                os.path.isdir(os.path.join(basePathToSoundFiles, name))]

    def generate_training_and_test_set(self):
        training_files = []
        test_files = []

        genres = get_genres_from_data()
        for genre in genres:
            directory_path = os.path.join(basePathToSoundFiles, genre, 'processed')

            if not os.path.exists(directory_path):
                continue

            processed_sound_files = os.listdir(directory_path)

            # maps file name pop.wav.npz to a relative path like ./data/genres/pop
            processed_sound_files = [os.path.join(directory_path, file) for file in processed_sound_files]

            np.random.shuffle(processed_sound_files)

            # 90% go into training set, 10% into test set
            split_index = round(len(processed_sound_files) * trainingTestSetRatio)
            training_files.extend(processed_sound_files[:split_index])
            test_files.extend(processed_sound_files[split_index:])

            target_value = next(v.get_target_value() for v in TARGET_VALUE_MAP if v.get_name() == genre)

            self.training_set.extend(self.__generate_prediction_sets_based_on_file_list(training_files, target_value))
            self.test_set.extend(self.__generate_prediction_sets_based_on_file_list(test_files, target_value))

        self.__create_training_file()
        self.__create_test_file()

    def __create_training_file(self):
        np.random.shuffle(self.training_set)
        save_to_numpy_file(os.path.join(basePathToSoundFiles, 'trainingSet.npz'), self.training_set)

    def __create_test_file(self):
        np.random.shuffle(self.test_set)
        save_to_numpy_file(os.path.join(basePathToSoundFiles, 'testSet.npz'), self.test_set)

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
            predicting_value = tuple([processed_song[index:index + frames_needed_for_input], target_prediction])
            set_of_prediction_values.append(predicting_value)
            index += int(1 / 2 * frames_needed_for_input)

        return set_of_prediction_values
