import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# Carica i dati di addestramento
train_features = np.load("train_features.npy")
labels = np.load("labels.npy")

# Lunghezza di ogni singolo segnale ECG
input_shape = train_features.shape[2]

# Crea il modello LSTM
model = Sequential()
model.add(LSTM(128, input_shape=(2, input_shape)))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))  #3 classi di output

#Bilancia le classi
class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)

# Compila il modello
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], class_weight=class_weights)

# Addestra il modello
model.fit(train_features, labels, epochs=50, batch_size=64, validation_split=0.2)

# Salva il modello allenato
model.save('lstm_model.keras')