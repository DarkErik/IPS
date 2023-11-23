import os
import tensorflow as tf

import AgeNNModel
import DataLoader
import Plotter
import TrainNeuralNetwork
import numpy as np
import constances

import website

PREPROCESSED_DATASET_FOLDER = os.path.join("data", "preprocessed")
UTK_FACE_FOLDER = os.path.join("data", "UTKFace")
MOOD_FACE_FOLDER = os.path.join("data", "MoodFaces")
UTK_DATASET_NAME = "utk"
AGE_EXTENSION = "age"
GENDER_EXTENSION = "gender"
RACE_EXTENSION = "race"
MOOD_EXTENSION = "mood"

PREPROCESS_DATA = False
TRAIN_OR_LOAD = "train"  # train or load or test
CURRENT_NETWORK = MOOD_EXTENSION

CKPT_TO_LOAD = "oldest"  # oldest or number
EPOCHS = 65


def main():
    print("Start Flask")
    website.run()
    exit(1)

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    if PREPROCESS_DATA:
        DataLoader.preprocess_UTK_images(UTK_FACE_FOLDER, PREPROCESSED_DATASET_FOLDER, UTK_DATASET_NAME)

    if TRAIN_OR_LOAD == "train":
        train_model()
    elif TRAIN_OR_LOAD == "load":
        train_ds, val_ds, test_ds = DataLoader.get_batched_datasets()
        evaluate_model(val_ds)
    else:
        test_models()





def evaluate_model(data_set):
    model = DataLoader.load_current_model(CURRENT_NETWORK)

    Plotter.get_evaluation_for_currnet_model(data_set, model, CURRENT_NETWORK)


def train_model():
    train_ds, val_ds, test_ds = DataLoader.get_batched_datasets(CURRENT_NETWORK)

    history, model = TrainNeuralNetwork.train_network(DataLoader.get_model_current_model(CURRENT_NETWORK),
                                                      EPOCHS,
                                                      train_ds,
                                                      val_ds,
                                                      CURRENT_NETWORK
                                                      )

    Plotter.get_evaluation_for_currnet_model(val_ds, model, CURRENT_NETWORK)
    Plotter.plot_history(history)


def test_models():
    global CURRENT_NETWORK

    testDir = os.path.join("data", "TestData")

    testDataLabels = os.listdir(testDir)

    testDataPxls = np.array([0.0] * (len(testDataLabels) * DataLoader.WIDTH * DataLoader.HEIGHT * 3))
    testDataPxls = testDataPxls.reshape((-1, DataLoader.WIDTH, DataLoader.HEIGHT, 3))

    index = 0
    for img in testDataLabels:
        testDataPxls[index] = DataLoader._preprocess_image(testDir, img, False)
        index += 1

    testDataPxls = testDataPxls.reshape((-1, DataLoader.WIDTH, DataLoader.HEIGHT, 3))

    model = DataLoader.load_current_model(AGE_EXTENSION)

    predictions = model.predict(x=testDataPxls, batch_size=1)
    predictions = np.argmax(predictions, axis = 1)
    print("------- AGE PREDICTION --------")
    for i in range(len(predictions)):
        if predictions[i] > 0:
            print(f"{testDataLabels[i]}: Between {constances.AGE_CATEGORIES[predictions[i] - 1]} and {constances.AGE_CATEGORIES[predictions[i]]}years old.")
        else:
            print(f"{testDataLabels[i]}: Less {constances.AGE_CATEGORIES[predictions[i]]} years old")
    model = DataLoader.load_current_model(GENDER_EXTENSION)

    predictions = model.predict(x=testDataPxls, batch_size=1)

    print("------- GENDER PREDICTION --------")
    for i in range(len(predictions)):
        if int(predictions[i][0] + 0.5) >= 1:
            print(f"{testDataLabels[i]} is female ({predictions[i][0]:.2f})")
        else:
            print(f"{testDataLabels[i]} is male ({predictions[i][0]:.2f})")


if __name__ == '__main__':
    main()
