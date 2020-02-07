from preprocessing.SetGenerator import SetGenerator
from preprocessing.SongFilePreprocessing import SongFilePreprocessor

preprocessor = SongFilePreprocessor()
preprocessor.generate_stft_frames()


generator = SetGenerator()
generator.generate_training_and_test_set()