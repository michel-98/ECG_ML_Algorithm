import copy

import flwr as fl
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from tsaug import TimeWarp, Quantize, Drift, Reverse

from parameter import *


# Flower client, adapted from Pytorch quickstart example
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, x_train, y_train, x_val, y_val, cid, knn_model: KNeighborsClassifier) -> None:
        # Create model
        self.model = get_model(x_train)
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val
        self.cid = cid
        self.knn_model = knn_model

    def get_parameters(self, config):
        return self.model.get_weights()

    def get_data(self):
        return self.x_train, self.y_train

    def fit(self, parameters, config):
        # Here you can call the specific fit method based on your requirements/configurations.
        # For example:
        print("Client: " + str(self.cid))
        if True:
            print("_________________With defense - replacement -augmentation __________________")
            return self.fit_without_poisoned_features_with_data_augmentation(parameters)
        else:
            print("_________________Without defense__________________")
            return self.fit_base(parameters)

    def fit_without_poisoned_features(self, parameters):
        """
        Fit the model without poisoned features.

        Parameters:
        - parameters: Parameters to set for the model.

        Returns:
        - Updated weights of the model, length of the training data, and an empty dictionary.
        """
        # Predict using the pre-trained KNN model
        y_pred = self.knn_model.predict(self.x_train.reshape(self.x_train.shape[0], -1))

        # Identify non-poisoned features (assuming a binary flag or label to identify poisoned samples)
        non_poisoned_indices = np.where(y_pred == self.y_train)[0]

        # Filter out poisoned features
        clean_x_train = self.x_train[non_poisoned_indices]
        clean_y_pred = copy.deepcopy(y_pred[non_poisoned_indices])

        # Set model weights
        self.model.set_weights(parameters)
        # Fit the model using only the non-poisoned features
        self.model.fit(
            clean_x_train, clean_y_pred, epochs=1, batch_size=32, verbose=VERBOSE
        )
        self.y_train = clean_y_pred
        self.x_train = clean_x_train
        return self.model.get_weights(), len(clean_x_train), {}

    def fit_without_poisoned_features_with_data_augmentation(self, parameters):
        """
        Fit the model without poisoned features.

        Parameters:
        - parameters: Parameters to set for the model.

        Returns:
        - Updated weights of the model, length of the training data, and an empty dictionary.
        """
        # Predict using the pre-trained KNN model
        y_pred = self.knn_model.predict(self.x_train.reshape(self.x_train.shape[0], -1))

        # Identify non-poisoned features (assuming a binary flag or label to identify poisoned samples)
        non_poisoned_indices = np.where(y_pred == self.y_train)[0]
        if len(non_poisoned_indices) != len(y_pred):
            print(" Client: " + str(self.cid) + "  entrato in augmentation")
            # Filter out poisoned features
            clean_x_train = self.x_train[non_poisoned_indices]
            augmenter = (
                    TimeWarp() * 2 +
                    Quantize(n_levels=10) * 2 +
                    Drift(max_drift=(0.1, 0.5)) * 2 +
                    Reverse() * 2
            )
            augmented_x_train = augmenter.augment(clean_x_train)
            augmented_clean_y_pred = self.knn_model.predict(augmented_x_train.reshape(augmented_x_train.shape[0], -1))
            # Set model weights
            self.model.set_weights(parameters)
            # Fit the model using only the non-poisoned features
            self.model.fit(
                augmented_x_train, augmented_clean_y_pred, epochs=1, batch_size=32, verbose=VERBOSE
            )
            self.x_train = augmented_x_train
            self.y_train = augmented_clean_y_pred
            return self.model.get_weights(), len(self.y_train), {}
        else:
            return self.fit_base(parameters)

    def fit_replace(self, parameters):
        y_pred = self.knn_model.predict(self.x_train.reshape(self.x_train.shape[0], -1))
        # Set model weights
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train, y_pred, epochs=1, batch_size=32, verbose=VERBOSE
        )
        self.y_train = y_pred

        return self.model.get_weights(), len(self.y_train), {}

    def fit_base(self, parameters):
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train, self.y_train, epochs=1, batch_size=32, verbose=VERBOSE
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(
            self.x_val, self.y_val, batch_size=64, verbose=VERBOSE
        )
        return loss, len(self.x_val), {"accuracy": acc}
