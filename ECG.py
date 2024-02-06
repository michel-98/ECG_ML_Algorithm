import numpy as np
import pywt
import wfdb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self, utenti):
        """Initialize DataPreprocessor object."""
        self.utenti = utenti
        self.classi_aritmia = ['e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F']
        self.dataset_folder = "MIT-BIH-DataSet"
        self.train_features = []
        self.labels = []
        self.process_data()

    def process_data(self):
        """Process ECG data for each user in utenti list."""
        for utente in self.utenti:
            utente = str(utente)
            percorso_file = f"{self.dataset_folder}/{utente}.dat"
            annotation = wfdb.rdann(f"{self.dataset_folder}/{utente}", 'atr')

            for i in range(1, len(annotation.symbol) - 1):
                if annotation.symbol[i] in self.classi_aritmia:
                    sampfrom = annotation.sample[i - 1] + int((annotation.sample[i] - annotation.sample[i - 1]) / 2)
                    sampto = annotation.sample[i] + int((annotation.sample[i + 1] - annotation.sample[i]) / 2)
                    record = wfdb.rdrecord(percorso_file.split('.')[0], sampfrom=sampfrom, sampto=sampto, channels=[0])
                    record = record.p_signal.reshape(-1)
                    lunghezza_target = 280
                    record_interpolato = np.interp(
                        np.linspace(0, 1, lunghezza_target),
                        np.linspace(0, 1, len(record)),
                        record
                    )
                    cwtmatr, freqs = pywt.cwt(record_interpolato, 1000, 'mexh')
                    feature = []
                    feature.append(record_interpolato)
                    feature.append(cwtmatr.ravel())

                    if annotation.symbol[i] in ['e', 'j', 'A', 'a', 'J', 'S']:
                        label = 0
                    elif annotation.symbol[i] in ['V', 'E']:
                        label = 1
                    elif annotation.symbol[i] == 'F':
                        label = 2
                    else:
                        label = 3

                    self.train_features.append(feature)
                    self.labels.append(label)

        self.train_features = np.asarray(self.train_features)
        self.labels = np.asarray(self.labels)

    def split_data(self):
        """Split data into training and testing sets, and apply SMOTE on the training set."""
        # Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(
            self.train_features, self.labels, test_size=0.3, random_state=42
        )

        # Apply SMOTE only on the training set
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        x_train_resampled, y_train_resampled = smote.fit_resample(x_train.reshape(x_train.shape[0], -1), y_train)
        x_train_resampled = x_train_resampled.reshape(x_train_resampled.shape[0], 2, 280)

        return x_train_resampled, y_train_resampled

    def get_test_data(self):
        return self.train_features, self.labels
