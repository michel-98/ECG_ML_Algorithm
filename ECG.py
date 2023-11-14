import pywt
import wfdb
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
import os
# Definisci le classi di aritmia

classi_aritmia = ['e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F']

 # Cartella di destinazione per le immagini
cartella_destinazione = 'immagini_aritmia'
#os.makedirs(cartella_destinazione, exist_ok=True)
# Utenti del dataset MIT-BIH Arrhythmia
utenti = ['100', '101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230']

# Lista per le features complessive
train_features = []
labels = []

for utente in utenti:
    percorso_file = utente + '.dat'
    annotation = wfdb.rdann(utente, 'atr')
    sampfrom = 0
 # Cartella specifica per l'utente
    cartella_utente = os.path.join(cartella_destinazione, utente)
    #os.makedirs(cartella_utente, exist_ok=True)

    for i in range(1, len(annotation.symbol) - 1):
        if annotation.symbol[i] in classi_aritmia:
            # Esegui l'interpolazione lineare
            features_interpolati = []  # Lista per le features specifiche di questo battito
            sampfrom = annotation.sample[i - 1] + int((annotation.sample[i] - annotation.sample[i - 1]) / 2)
            sampto = annotation.sample[i] + int((annotation.sample[i + 1] - annotation.sample[i]) / 2)
            record = wfdb.rdrecord(percorso_file.split('.')[0], sampfrom=sampfrom, sampto=sampto, channels=[0])
            record = record.p_signal.reshape(-1)
            # Esegui l'interpolazione lineare sul record e calcola cwtmatr su di esso
            lunghezza_target = 280
            record_interpolato = np.interp(
            np.linspace(0, 1, lunghezza_target),
            np.linspace(0, 1, len(record)),
            record
            )
            cwtmatr, freqs = pywt.cwt(record_interpolato, 1000, 'mexh')
            # Unisci le diverse feature in un'unica rappresentazione per ogni osservazione
            feature = []
            feature.append(record_interpolato)
            feature.append(cwtmatr.ravel())
              # Assegna un'etichetta numerica in base alla classe
            if annotation.symbol[i] in ['e', 'j', 'A', 'a', 'J', 'S']:
                label = 0
            elif annotation.symbol[i] in ['V', 'E']:
                label = 1
            elif annotation.symbol[i] == 'F':
                label = 2
            else:
                label = 3  # Etichetta per le altre classi
            train_features.append(feature)
            labels.append(label)

# Converte le liste in array NumPy
train_features = np.asarray(train_features)
labels = np.asarray(labels)

# Applica SMOTE
train_features = train_features.reshape(train_features.shape[0], -1)
smote = SMOTE(sampling_strategy='auto', random_state=42)
train_features_resampled, labels_resampled = smote.fit_resample(train_features, labels)
train_features_resampled = train_features_resampled.reshape(train_features_resampled.shape[0], 2, lunghezza_target)

# Divisione dei dati in set di addestramento e test (70% addestramento, 30% test)
features_train, features_test, labels_train, labels_test = train_test_split(features_resampled, labels_resampled, test_size=0.3, random_state=42)


# Adesso train_features_resampled e labels_resampled contengono le classi bilanciate
# Calcola il conteggio delle etichette per ciascuna classe
conteggio_classi = np.bincount(labels_resampled)

# Stampa il conteggio delle etichette per ciascuna classe
for classe, conteggio in enumerate(conteggio_classi):
    print(f"Classe {classe}: {conteggio} etichette")

# Salva i dati
np.save("train_features.npy", features_train)
np.save("train_labels.npy", labels_train)
np.save("test_features.npy", features_test)
np.save("test_labels.npy", labels_test)