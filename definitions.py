import os
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

ROOT_DIR: str = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
main_activity_label_classes: dict[str, int] = {
    'label:UNKNOWN': 0,
    'label:LYING_DOWN': 1,
    'label:SITTING': 2,
    'label:OR_standing': 3,
    'label:FIX_walking': 4,
    'label:FIX_running': 5,
    'label:BICYCLING': 6
}
batch_size: int = 32
window_size: int = 10
early_stopping_patience: int = 5
learning_rate: float = 0.00005
epochs: int = 100
n_y = len(main_activity_label_classes.keys())


# Source: https://stackoverflow.com/questions/43114460/is-there-a-way-to-reshape-an-array-that-does-not-maintain-the-original-size-or
def reshape_and_truncate(arr, shape):
    desired_size_factor = np.prod([n for n in shape if n != -1])
    if -1 in shape:  # implicit array size
        desired_size = arr.size // desired_size_factor * desired_size_factor
    else:
        desired_size = desired_size_factor
    return arr.flat[:desired_size].reshape(shape)


def encodeTarget(y):
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    return np_utils.to_categorical(y)


def get_labels(windows):
    predicted_labels = []

    for p in windows:
        df = pd.DataFrame(p)
        probabilities = []

        for col in df.columns:
            probabilities.append(df[col].mean())
        predicted_labels.append(np.argmax(probabilities))
    return predicted_labels


def plot_val_loss_progress(history, export_dir: str = 'Plots'):
    os.makedirs(export_dir, exist_ok=True)
    val_loss = history.history['val_loss']
    plt.plot(val_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss LSTM')
    plt.savefig(f'{export_dir}/convergence_val_loss_lstm.png')


def plot_conf_matrix(gold_labels, predicted_labels, export_dir: str = 'Plots'):
    os.makedirs(export_dir, exist_ok=True)
    plt.figure(figsize=(10, 10))
    plt.title('Confusion Matrix')
    sn.heatmap(
        pd.DataFrame(tf.math.confusion_matrix(gold_labels, predicted_labels), index=main_activity_label_classes.keys(),
                     columns=main_activity_label_classes.keys()), annot=True, fmt='d', cmap='hot')
    plt.savefig(f'{export_dir}/confusion_matrix_lstm.png')
