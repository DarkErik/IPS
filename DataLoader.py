from typing import Any
from PIL import Image
import numpy as np
import tensorflow as tf
import os

import Util

WIDTH = 200
HEIGHT = 200


def preprocess_UTK_images(utk_folder, resulting_folder, dataset_name) -> Any:
    """
    Preprocesses a dataset for future computations
    :param utk_folder: folder of UTK images
    :param resulting_folder: folder in which the results gonna be saved
    :param dataset_name: name of the dataset, for future matching
    :return: nothing
    """
    all_images = [f for f in os.listdir(utk_folder) if os.path.isfile(os.path.join(utk_folder, f))]
    data = np.array([])
    labels = np.array([])

    max = len(all_images) // 10
    for i in range(0, max):
        pxls, lbl = _preprocess_image(utk_folder, all_images[i])
        data = np.append(data, pxls)
        labels = np.append(labels, lbl)
        Util.printProgressBar(i, max)

    Util.printProgressBar(max, max)
    data = data.reshape((-1, WIDTH, HEIGHT, 3))

    np.save(os.path.join(resulting_folder, dataset_name + "_data.npy"), data)
    np.save(os.path.join(resulting_folder, dataset_name + "_labels.npy"), labels)


def load_dataset_from_preprocessed(preprocessed_folder, dataset_name):
    """
    Loads a Dataset from the disc.
    :param preprocessed_folder: Folder with preprocessed data
    :param dataset_name: Name of the Dataset (e.x utk)
    :return: the according Tensorflow dataset
    """
    data = np.load(os.path.join(preprocessed_folder, dataset_name + "_data.npy"))
    labels = np.load(os.path.join(preprocessed_folder, dataset_name + "_labels.npy"))

    return create_TF_dataset_from_npArr(data, labels)

def create_TF_dataset_from_npArr(data, labels):
    """
    Creates a dataset from the corresponding data and labels array
    :param data: npArray
    :param labels: npArray
    :return: the according Tensorflow Dataset, and the size of it
    """
    return tf.data.Dataset.from_tensor_slices((data, labels)), len(labels)

def split_dataset_into_train_val_test(ds, ds_size, train, test, val):
    train_size = int(train * ds_size)
    test_size = int(test * ds_size)

    ds = ds.shuffle(int(ds_size / 4), reshuffle_each_iteration = False)
    train_dataset = ds.take(train_size)
    test_dataset = ds.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    test_dataset = test_dataset.take(test_size)

    return train_dataset, test_dataset, val_dataset


def _preprocess_image(image_dir, image_name) -> Any:
    """
    Preprocesses an images.
    Steps:
        - To Grayscale
        - To [0,1] Float value
    :param image_dir: path to the directory
    :param image_name: name of the image file
    :return: a pixel array (flattend, WIDTH * HEIGHT) and  the corresponding label (int)
    """
    img = Image.open(os.path.join(image_dir, image_name))
    pxls = np.array(img, float)

    # for ix in range(0, WIDTH):
    #     for iy in range(0, HEIGHT):
    #         pxls[ix, iy] /= 255.0

    split_path = image_name.split("_")
    label_age = int(split_path[0])

    return pxls, label_age
