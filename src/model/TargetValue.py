import numpy as np
import os
from Helper import loadDataFromNumpyFile, saveToNumpyFile

basePathToSoundFiles = "./data/genres"

# 80% test set, 10% validation set, 10% test set
trainingTestSetRatio = 0.8


class TargetValue:
    def __init__(self, name, targetValue):
        self.name = name
        self.targetValue = targetValue

    def getTargetValue(self):
        return self.targetValue

    def getName(self):
        return self.name


TARGET_VALUE_MAP = [
    TargetValue('blues', 0),
    TargetValue('classical', 1),
    TargetValue('country', 2),
    TargetValue('disco', 3),
    TargetValue('hiphop', 4),
    TargetValue('jazz', 5),
    TargetValue('metal', 6),
    TargetValue('pop', 7),
    TargetValue('reggae', 8),
    TargetValue('rock', 9),
]
