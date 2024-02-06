import copy

import flwr as fl
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from tsaug import TimeWarp, Quantize, Drift, Reverse

from parameter import *


# Flower client, adapted from Pytorch quickstart example
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, x_train, y_train, x_val, y_val, cid, knn_model: KNeighborsClassifier, defense_strategy) -> None:
        # Create client
        self.model = get_model(x_train)
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val
        self.cid = cid
        self.knn_model = knn_model
        self.defense_strategy = defense_strategy

    def get_parameters(self, config):
        return self.model.get_weights()

    def get_data(self):
        return self.x_train, self.y_train

    def fit(self, parameters, config):
        """Fit the model based on defense strategy."""
        if self.defense_strategy == 0:
            print("_________________With no defense_________________")
            return self.fit_base(parameters)
        elif self.defense_strategy == 1:
            print("_________________With defense - replacement_________________")
            return self.fit_replace(parameters)
        elif self.defense_strategy == 2:
            print("_________________With defense - deletion_________________")
            return self.fit_base(parameters)
        elif self.defense_strategy == 3:
            print("_________________With defense - replacement + augmentation_________________")
            return self.fit_without_poisoned_features_with_data_augmentation(parameters)
        else:
            print("_________________With no defense - ERROR  __________________")
            return self.fit_base(parameters)

    def fit_without_poisoned_features(self, parameters):
        """
        Fit the model without poisoned features. (DELETION)

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
        Fit the model without poisoned features but with data augmentation. (DATA AUGMENTATION)

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
        """
        Fit the model with replacement defense strategy. (SUBSTITUTION)

        Parameters:
        - parameters: Parameters to set for the model.

        Returns:
        - Updated weights of the model, length of the training data, and an empty dictionary.
        """
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
