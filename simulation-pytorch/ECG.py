import numpy as np
import pywt
import wfdb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def preprocess_data():
    classi_aritmia = ['e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', 'N', 'L', 'R']
    cartella_destinazione = 'immagini_aritmia'
    utenti = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
              '116', '117', '118', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
              '209', '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230', '231',
              '232', '233', '234']

    dataset_folder = "MIT-BIH-DataSet"
    train_features = []
    labels = []
    for utente in utenti:
        percorso_file = dataset_folder + '/' + utente + '.dat'
        annotation = wfdb.rdann(dataset_folder + '/' + utente, 'atr')

        for i in range(1, len(annotation.symbol) - 1):
            if annotation.symbol[i] in classi_aritmia:
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

                train_features.append(feature)
                labels.append(label)

    train_features = np.asarray(train_features)
    labels = np.asarray(labels)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(train_features, labels, test_size=0.3, random_state=42)

    # Apply SMOTE only on the training set
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train.reshape(x_train.shape[0], -1), y_train)
    x_train_resampled = x_train_resampled.reshape(x_train_resampled.shape[0], 2, lunghezza_target)

    return x_train_resampled, y_train_resampled, (x_test, y_test)
