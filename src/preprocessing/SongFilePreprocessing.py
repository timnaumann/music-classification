import numpy as np
from scipy import signal
from scipy.io import wavfile
import os
import math

from preprocessing.SetGenerator import SetGenerator
from Helper import save_to_numpy_file, is_wav_file

basePathToSoundFiles = "./data/genres"
processedFolder = 'processed'


def split_signal_into_frames(signal_values, sample_rate):
    frames = []

    # split into 23ms frames
    amount_of_value_per_frame = math.ceil((sample_rate / 1000) * 23)
    current_frame = 0
    start_index = 0

    while start_index <= signal_values.size:
        end_index = start_index + amount_of_value_per_frame

        frames.append(np.array(signal_values[start_index:end_index]))

        current_frame += 1
        start_index = int(current_frame * amount_of_value_per_frame)

    return np.array(frames)

def generate_stft_frames():
    genres = [name for name in os.listdir(basePathToSoundFiles) if os.path.isdir(basePathToSoundFiles + "/" + name)]

    for directory in genres:
        wav_file_directory_path = os.path.join(basePathToSoundFiles, directory)
        sound_files = os.listdir(wav_file_directory_path)
        for file in sound_files:
            if is_wav_file(file):
                frame_transformation_values = []
                sample_rate, signal_values = wavfile.read(os.path.join(wav_file_directory_path, file))
                frames_of_signal = split_signal_into_frames(signal_values, sample_rate)
                for frame in frames_of_signal:
                    f, t, Zxx = signal.stft(frame)
                    frame_transformation_values.append(Zxx)

                processed_folder_path = os.path.join(wav_file_directory_path, processedFolder)
                if not os.path.exists(processed_folder_path):
                    os.mkdir(processed_folder_path)
                    print("Directory ", processed_folder_path, " created")

                file_name_for_song = os.path.join(processed_folder_path, file + ".npz")
                save_to_numpy_file(file_name_for_song, frame_transformation_values)
                break
        break




