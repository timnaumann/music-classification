import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
import os
import math

from Helper import getListOfPreprocessedSongs, removeFiles, removeFiles

preprocessedSongs = getListOfPreprocessedSongs()

removeFiles(preprocessedSongs)
