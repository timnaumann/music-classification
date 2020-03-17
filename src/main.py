from preprocessing.SetGenerator import SetGenerator
from model.Model import Model
import Globals
from sklearn.model_selection import KFold

generator = SetGenerator()
test_set = generator.get_test_set()


def train_with_cross_validation(model):
    training_set = generator.get_training_set()
    for train_index, validation_index in KFold(5, shuffle=True).split(training_set):
        training_subset = training_set[train_index]
        validation_subset = training_set[validation_index]

        model.fit(training_subset, validation_subset, epochs=5)


# create model and train it
model = Model(Globals.MODEL_VERSION_1)
train_with_cross_validation(model)

# if you want to load an already trained model use
# model.load()

print(model.evaluate(test_set))
