import argparse
import os
from datetime import datetime

from keras import Sequential
from keras.src.layers import LSTM, Dense

from parameter import *

timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f"path/to/your/logs/{timestamp_str}"

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

parser = argparse.ArgumentParser(description="Flower Simulation with Tensorflow/Keras")

parser.add_argument(
    "--num_cpus",
    type=int,
    default=1,
    help="Number of CPUs to assign to a virtual client",
)
parser.add_argument(
    "--num_gpus",
    type=float,
    default=0.0,
    help="Ratio of GPU memory to assign to a virtual client",
)
parser.add_argument("--num_rounds", type=int, default=10, help="Number of FL rounds.")

NUM_CLIENTS = 10
VERBOSE = 0


def get_model(train_features):
    """Constructs a simple model architecture suitable for MNIST."""

    input_shape = train_features.shape[1]

    model = Sequential()
    model.add(LSTM(128, input_shape=(2, input_shape)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # 3 classi di output - prova sigmoid

    # Bilancia le classi
    # DO not work see it later  class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)

    # Compila il modello
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
