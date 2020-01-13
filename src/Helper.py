import numpy as np
import os

basePathToSoundFiles = "./data/genres"

# 80% test set, 10% validation set, 10% test set
trainingTestSetRatio = 0.8

def getListOfPreprocessedSongs():
    listOfFiles = []
    genres = getGenresFromData()
    for directory in genres:
        directoryPath = os.path.join(basePathToSoundFiles,directory)
        soundFiles = os.listdir(directoryPath)
        for file in soundFiles:
            if isPreprocessedSongFile(file):
                listOfFiles.append(os.path.join(directoryPath, file))
    return listOfFiles



def removeFiles(files):
    for file in files:
        os.remove(file)

def saveToNumpyFile(fileName, values):
    np.savez(fileName, values)

def loadDataFromNumpyFile(fileName):
    container = np.load(fileName, allow_pickle=True)
    return [container[key] for key in container]

def isWavFile(file):
    return file.endswith('.wav')

def isPreprocessedSongFile(file):
    return file.endswith('.npz')

def getGenresFromData():
    return [name for name in os.listdir(basePathToSoundFiles) if os.path.isdir(os.path.join(basePathToSoundFiles,name))]
