import tensorflow as tf
from typing import Any


NETWORK_SAVE_PATH_BASE_FOLDER = "\\trainedNetworks"
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
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(
        trainings_data,
        validation_data=validation_data,
        epochs=epochs
    )

    return history


def get_checkpoint_callback():
    checkpoint_path = f"{NETWORK_SAVE_PATH_BASE_FOLDER}\\{_network_save_path_addition}\\epoche-{_current_training_epoch}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    return cp_callback

