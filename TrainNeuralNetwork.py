import os

import tensorflow as tf
from typing import Any

import AgeNNModel
import GenderNNModel
import main


NETWORK_SAVE_PATH_BASE_FOLDER = "trainedNetworks"
_network_save_path_addition = ""
_current_training_epoch: int = 0

def train_network(model, epochs, trainings_data, validation_data, network_save_path_addition) -> Any:
    """
    Trains a network model for certain epochs
    Saves weights after every trainingsepche to a file for further access.

    Returns the trainings history
    """
    global _network_save_path_addition
    global _current_training_epoch

    _current_training_epoch = 0
    _network_save_path_addition = network_save_path_addition

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'],
                  )

    model.summary()

    history = model.fit(
        trainings_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks = [get_checkpoint_callback()]
    )

    return history, model


# def get_checkpoint_callback():
#     global _current_training_epoch
#     checkpoint_path = os.path.join(NETWORK_SAVE_PATH_BASE_FOLDER, _network_save_path_addition, f"epoche-{_current_training_epoch}.ckpt")
#     _current_training_epoch = 1
#     cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                      save_weights_only=True,
#                                                      verbose=1)
#     return cp_callback

def get_checkpoint_callback():
    checkpoint_path = os.path.join(NETWORK_SAVE_PATH_BASE_FOLDER, _network_save_path_addition, "epoche-{epoch:04d}.ckpt");
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True)
    return cp_callback


def get_model_current_model():
    if main.CURRENT_NETWORK == main.AGE_EXTENSION:
        return AgeNNModel.getModel()
    elif main.CURRENT_NETWORK == main.GENDER_EXTENSION:
        return GenderNNModel.getModel()
    else:
        print("UNKOWN MODEL IN main.CURRENT_NETWORK")