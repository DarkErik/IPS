import os

import AgeNNModel
import DataLoader
import Plotter
import TrainNeuralNetwork

PREPROCESSED_DATASET_FOLDER = os.path.join("data", "preprocessed")
UTK_FACE_FOLDER = os.path.join("data", "UTKFace")
UTK_DATASET_NAME = "utk"
AGE_EXTENSION = "age"
GENDER_EXTENSION = "gender"
RACE_EXTENSION = "race"

PREPROCESS_DATA = False
TRAIN_OR_LOAD = "train" #train or load
CURRENT_NETWORK = AGE_EXTENSION

CKPT_TO_LOAD = "oldest" #oldest or number
EPOCHS = 10

def main():
    if PREPROCESS_DATA:
        DataLoader.preprocess_UTK_images(UTK_FACE_FOLDER, PREPROCESSED_DATASET_FOLDER, UTK_DATASET_NAME)

    if TRAIN_OR_LOAD == "train":
        train_model()
    else:
        evaluate_model()



def get_batched_datasets():
    dataset, dataset_size = DataLoader.load_dataset_from_preprocessed(PREPROCESSED_DATASET_FOLDER, UTK_DATASET_NAME, CURRENT_NETWORK)
    train_ds, test_ds, val_ds = DataLoader.split_dataset_into_train_val_test(dataset, dataset_size, 0.80, 0.1, 0.1)

    train_ds = train_ds.batch(16)
    test_ds = test_ds.batch(16)
    val_ds = val_ds.batch(16)

    return train_ds, val_ds, test_ds


def evaluate_model():
    train_ds, val_ds, test_ds = get_batched_datasets()


    model = DataLoader.load_current_model()

    Plotter.get_evaluation_for_currnet_model(val_ds, model)


def train_model():
    train_ds, val_ds, test_ds = get_batched_datasets()

    history, model = TrainNeuralNetwork.train_network(TrainNeuralNetwork.get_model_current_model(),
                                                      EPOCHS,
                                                      train_ds,
                                                      val_ds,
                                                      CURRENT_NETWORK
                                                      )

    Plotter.get_evaluation_for_currnet_model(val_ds, model)
    Plotter.plot_history(history)

if __name__ == '__main__':
    main()