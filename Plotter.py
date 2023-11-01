from typing import Any

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


def log_prediction_of_age(predict_data, model):
    texts = np.array([])
    text_labels = np.array([])
    for x, y in predict_data.take(-1):
        texts = np.append(texts, x.numpy())
        text_labels = np.append(text_labels, y.numpy())

    texts = texts.reshape((-1, 200, 200, 3));
    predictions = model.predict(x=texts, batch_size=1)
    for i in range(len(predictions)):
        print(f"{i}:\t{text_labels[i]}\t{predictions[i][0]}\tYears off: {abs(text_labels[i] - predictions[i][0])}")
def plot_confusion_matrix_raw(text_labels, predictions, output_neurons, normalize = False):
    if normalize:
        normalize = 'true'
    else:
        normalize = None
    # predictions = [int(p) for p in predictions]
    cm = confusion_matrix(text_labels, predictions, normalize=normalize)  # np.argmax(text_labels, axis=1)
    # cm = confusion_matrix(predictions, text_labels)  # =1

    cm_df = pd.DataFrame(cm,
                         index=[i for i in range(output_neurons)],
                         columns=[i for i in range(output_neurons)])

    plt.figure(figsize=(5, 4))

    fmt = '.4g'
    if normalize:
        fmt = '.2g'

    sns.heatmap(cm_df, annot=True, fmt = fmt)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')

    plt.show()


def plot_confusion_matrix(predict_data, model, output_neurons, normalize=False, both_cms=False):
    texts = np.array([])
    text_labels = np.array([])
    for x, y in predict_data.take(-1):
        texts = np.append(texts, x.numpy())
        text_labels = np.append(text_labels, y.numpy())

    predictions = model.predict(x=texts, batch_size=1)
    model.evaluate(predict_data)

    predictions = np.argmax(predictions, axis=1)
    text_labels = text_labels.reshape((-1, output_neurons))
    text_labels = np.argmax(text_labels, axis=1)

    if both_cms:
        plot_confusion_matrix_raw(text_labels, predictions, False)
        plot_confusion_matrix_raw(text_labels, predictions, True)
    else:
        plot_confusion_matrix_raw(text_labels, predictions, normalize)


def plot_history(histroy: Any) -> None:
    plt.plot(histroy.history["accuracy"], label='Accuracy')
    plt.plot(histroy.history["val_accuracy"], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.ylabel('%')
    plt.xlabel('Epoch')
    plt.legend(loc="upper left")
    plt.show()

    plt.plot(histroy.history["loss"], label='Loss')
    plt.plot(histroy.history["val_loss"], label='Validation Loss')
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc="upper left")
    plt.show()
