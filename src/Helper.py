import numpy as np
import os

import Globals

def get_list_of_preprocessed_songs():
    listOfFiles = []
    genres = get_genres_from_data()
    for directory in genres:
        directoryPath = os.path.join(Globals.BASE_PATH_TO_SOUND_FILES, directory)
        soundFiles = os.listdir(directoryPath)
        for file in soundFiles:
            if is_preprocessed_song_file(file):
                listOfFiles.append(os.path.join(directoryPath, file))
    return listOfFiles


def remove_files(files):
    for file in files:
        os.remove(file)


def save_to_numpy_file(fileName, values):
    np.savez(fileName, values)


def load_data_from_numpy_file(fileName):
    container = np.load(fileName, allow_pickle=True)
    return [container[key] for key in container]


def is_wav_file(file):
    return file.endswith('.wav')


def is_preprocessed_song_file(file):
    return file.endswith('.npz')


def get_genres_from_data():
    return [name for name in os.listdir(Globals.BASE_PATH_TO_SOUND_FILES) if
            os.path.isdir(os.path.join(Globals.BASE_PATH_TO_SOUND_FILES, name))]
