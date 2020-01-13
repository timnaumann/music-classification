import numpy as np
import os
from Helper import loadDataFromNumpyFile, saveToNumpyFile
from model.TargetValue import TARGET_VALUE_MAP

basePathToSoundFiles = "./data/genres"

# 90% test set, 10% test set
trainingTestSetRatio = 0.9

class SetGenerator:
    def __init__(self):
        self.testSet = []
        self.trainingSet = []

    def generateTrainingAndTestSet(self):
        trainingFiles = []
        testFiles = []

        genres = self.getGenresFromData()
        for genre in genres:
            directoryPath = os.path.join(basePathToSoundFiles, genre, 'processed')
            processedSoundFiles = os.listdir(directoryPath)
            np.random.shuffle(processedSoundFiles)

            # 90% go into training set, 10% into test set
            splitIndex = round(len(processedSoundFiles) * trainingTestSetRatio)
            trainingFiles.append(processedSoundFiles[:splitIndex])
            testFiles.append(processedSoundFiles[splitIndex:])

            targetValue = next(v.getTargetValue() for v in TARGET_VALUE_MAP if v.getName() == genre)

            self.trainingSet.append(self.generatePredictionSetsBasedOnFileList(trainingFiles, targetValue))
            self.testSet.append(self.generatePredictionSetsBasedOnFileList(testFiles, targetValue))

        self.createTrainingFile()
        self.createTestFile()


    def createTrainingFile(self):
        np.random.shuffle(self.trainingSet)
        saveToNumpyFile(os.path.join(basePathToSoundFiles, 'trainingSet.npz'), self.trainingSet)

    def createTestFile(self):
        np.random.shuffle(self.testSet)
        saveToNumpyFile(os.path.join(basePathToSoundFiles, 'testSet.npz'),  self.testSet)

    def generatePredictionSetsBasedOnFileList(self, files, genre):
        resultingFile = []
        for file in files:
            resultingFile.append(self.extractValuesFromFile(file, genre))
        return resultingFile


    # output is a list of the shape [( 128 x 513 tensor, targetValue), ...]
    def extractValuesFromFile(self, filePath, targetPrediction):
        processedSong = loadDataFromNumpyFile(filePath)

        setOfPredictionValues = []

        framesNeededForInput = 128
        amountOfFrames = len(processedSong)
        index = 0
        while(index <= amountOfFrames):
            predictingValue = tuple([processedSong[index + framesNeededForInput], targetPrediction])
            setOfPredictionValues.append(predictingValue)

            index += 1/2 * framesNeededForInput
        return setOfPredictionValues


    def getGenresFromData(self):
        return [name for name in os.listdir(basePathToSoundFiles) if
                os.path.isdir(os.path.join(basePathToSoundFiles, name))]
