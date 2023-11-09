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


def main():
    if PREPROCESS_DATA:
        DataLoader.preprocess_UTK_images(UTK_FACE_FOLDER, PREPROCESSED_DATASET_FOLDER, UTK_DATASET_NAME)

    dataset, dataset_size = DataLoader.load_dataset_from_preprocessed(PREPROCESSED_DATASET_FOLDER, UTK_DATASET_NAME)
    train_ds, test_ds, val_ds = DataLoader.split_dataset_into_train_val_test(dataset, dataset_size, 0.80, 0.1, 0.1)

    train_ds = train_ds.batch(16)
    test_ds = test_ds.batch(16)
    val_ds = val_ds.batch(16)

    history, model = TrainNeuralNetwork.train_network(AgeNNModel.getModel(),
                                     30,
                                     train_ds,
                                     val_ds,
                                     AGE_EXTENSION
                                     )

    Plotter.log_prediction_of_age(val_ds, model);

if __name__ == '__main__':
    main()