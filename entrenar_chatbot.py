# Importación de librerías necesarias
import json
import numpy as np
import random
import nltk
import pickle


# Librerías de NLP y Keras para el modelo
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('wordnet')

# Inicializar lematizador
lemmatizer = WordNetLemmatizer()

# Cargar dataset archivo JSON con los datos del chatbot
with open("compostaje_dataset.json", encoding='utf-8') as file:
    data = json.load(file)


# Listas para palabras, clases y documentos de entrenamiento
words = []
classes = []
documents = []
ignore_letters = ['?', '¿', '!', '.', ',']

# Procesamiento de patrones de texto en el dataset
for intent in data['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lematizar y limpiar las palabras
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))


# Preparación del conjunto de entrenamiento
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]
    for w in words:
        bag.append(1 if w in pattern_words else 0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Mezclar los datos y convertir a numpy
random.shuffle(training)
training = np.array(training, dtype=object)


# Separar entradas y salidas
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))


# Crear la red neuronal (modelo secuencial)
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compilar el modelo con el optimizador SGD
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenar el modelo con los datos preparados
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Guardar el modelo entrenado y las estructuras necesarias
model.save("chatbot_model.keras") # Guardar el modelo
pickle.dump(words, open('words.pkl', 'wb')) # Guardar vocabulario
pickle.dump(classes, open('classes.pkl', 'wb'))  # Guardar clases

print("Modelo entrenado y archivos guardados.")

