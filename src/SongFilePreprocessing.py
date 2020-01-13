import tensorflow as tf
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
import os
import math

from Helper import loadDataFromNumpyFile, saveToNumpyFile, isWavFile

basePathToSoundFiles = "./data/genres"

genres = [name for name in os.listdir(basePathToSoundFiles) if os.path.isdir(basePathToSoundFiles + "/" + name)]


def splitSignalIntoFrames(signalValues, sampleRate):
    frames = []

    # split into 23ms frames
    amountOfValuePerFrame = math.ceil((sampleRate / 1000) * 23)
    currentFrame = 0
    startIndex = 0

    while startIndex <= signalValues.size:
        endIndex = startIndex + amountOfValuePerFrame

        frames.append(np.array(signalValues[startIndex:endIndex]))

        currentFrame += 1
        startIndex = int(currentFrame * amountOfValuePerFrame)

    return np.array(frames)

processedFolder = 'processed'

for directory in genres:
    wavFileDirectoryPath = os.path.join(basePathToSoundFiles, directory)
    soundFiles = os.listdir(wavFileDirectoryPath)
    for file in soundFiles:
        if isWavFile(file):
            frameTransformationValues = []
            sampleRate, signalValues = wavfile.read(os.path.join(wavFileDirectoryPath, file))
            framesOfSignal = splitSignalIntoFrames(signalValues, sampleRate)
            for frame in framesOfSignal:
                f, t, Zxx = signal.stft(frame)
                frameTransformationValues.append(Zxx)

            processedFolderPath = os.path.join(wavFileDirectoryPath, processedFolder)

            if not os.path.exists(processedFolderPath):
                os.mkdir(processedFolderPath)
                print("Directory ", processedFolderPath, " created")

            fileNameForSong = os.path.join(processedFolderPath, file + ".npz")
            saveToNumpyFile(fileNameForSong, frameTransformationValues)

            print(len(frameTransformationValues))
            break
    break


