import numpy as np
import pywt
import wfdb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from parameter import *


def preprocess_data():
    # Define arrhythmia classes
    global lunghezza_target
    classi_aritmia = ['e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F']

    # Destination folder for images
    cartella_destinazione = 'immagini_aritmia'
    # os.makedirs(cartella_destinazione, exist_ok=True)
    # MIT-BIH Arrhythmia dataset users
    utenti = ['100', '101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122', '124', '201', '203',
              '205', '207', '208', '209', '215', '220', '223', '230']

    dataset_folder = "MIT-BIH-DataSet"
    # Lists for overall features
    train_features = []
    labels = []

    for utente in utenti:
        percorso_file = dataset_folder + '/' + utente + '.dat'
        annotation = wfdb.rdann(dataset_folder + '/' + utente, 'atr')
        sampfrom = 0
        # Specific folder for the user
        cartella_utente = os.path.join(cartella_destinazione, utente)
        # os.makedirs(cartella_utente, exist_ok=True)

        for i in range(1, len(annotation.symbol) - 1):
            if annotation.symbol[i] in classi_aritmia:
                # Linear interpolation
                features_interpolati = []  # List for specific features of this beat
                sampfrom = annotation.sample[i - 1] + int((annotation.sample[i] - annotation.sample[i - 1]) / 2)
                sampto = annotation.sample[i] + int((annotation.sample[i + 1] - annotation.sample[i]) / 2)
                record = wfdb.rdrecord(percorso_file.split('.')[0], sampfrom=sampfrom, sampto=sampto, channels=[0])
                record = record.p_signal.reshape(-1)
                # Linear interpolation on the record and calculate cwtmatr on it
                lunghezza_target = 280
                record_interpolato = np.interp(
                    np.linspace(0, 1, lunghezza_target),
                    np.linspace(0, 1, len(record)),
                    record
                )
                cwtmatr, freqs = pywt.cwt(record_interpolato, 1000, 'mexh')
                # Merge different features into a single representation for each observation
                feature = [record_interpolato, cwtmatr.ravel()]
                # Assign a numerical label based on the class
                if annotation.symbol[i] in ['e', 'j', 'A', 'a', 'J', 'S']:
                    label = 0
                elif annotation.symbol[i] in ['V', 'E']:
                    label = 1
                elif annotation.symbol[i] == 'F':
                    label = 2
                else:
                    label = 3  # Label for other classes
                train_features.append(feature)
                labels.append(label)

    # Convert lists to NumPy arrays
    train_features = np.asarray(train_features)
    labels = np.asarray(labels)

    # Applica SMOTE
    train_features = train_features.reshape(train_features.shape[0], -1)
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    train_features_resampled, labels_resampled = smote.fit_resample(train_features, labels)
    train_features_resampled = train_features_resampled.reshape(train_features_resampled.shape[0], 2, lunghezza_target)

    # Now train_features_resampled and labels_resampled contain balanced classes
    # Calculate label counts for each class
    conteggio_classi = np.bincount(labels_resampled)

    # Print label counts for each class
    for classe, conteggio in enumerate(conteggio_classi):
        print(f"Classe {classe}: {conteggio} etichette")

    train_features, test_feature = train_test_split(train_features_resampled, test_size=0.3, random_state=42)
    split_train_features = np.array_split(train_features_resampled, NUM_CLIENTS, axis=0)


    # Save data
    np.save("train_features.npy", train_features_resampled)
    np.save("test_features.npy", test_feature)

    return split_train_features, test_feature,
