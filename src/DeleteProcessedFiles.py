import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
import os
import math

from Helper import get_list_of_preprocessed_songs, remove_files, remove_files

preprocessedSongs = get_list_of_preprocessed_songs()

remove_files(preprocessedSongs)
