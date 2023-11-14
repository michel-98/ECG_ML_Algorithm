import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

# Caricamento del modello precedentemente addestrato
model = load_model('lstm_model.keras')

# Caricamento dei dati di test
test_features = np.load("test_features.npy")
test_labels = np.load("test_labels.npy")

# Valutazione del modello sui dati di test
loss, accuracy = model.evaluate(test_features, test_labels)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Predizioni su tutti i dati di test
predictions = model.predict(test_features)

# Converti le predizioni in classi
predicted_classes = np.argmax(predictions, axis=1)

# Calcola la matrice di confusione
confusion_mat = confusion_matrix(test_labels, predicted_classes)

# Visualizza la matrice di confusione
print("\nMatrice di Confusione:")
print(confusion_mat)

# Visualizza il report di classificazione
print("\nReport di Classificazione:")
print(classification_report(test_labels, predicted_classes))
