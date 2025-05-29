from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Montar la carpeta 'static' para servir archivos estáticos como CSS e imágenes
app.mount("/static", StaticFiles(directory="static"), name="static")

# Carpeta donde están los HTML
templates = Jinja2Templates(directory="templates")

# Ruta principal ("/") que muestra la página de inicio con un formulario para ingresar nombre
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Esto redirige al menú con el nombre del usuario
@app.get("/menu", response_class=HTMLResponse)
async def menu(request: Request, nombre: str = ""):
    return templates.TemplateResponse("menu.html", {"request": request, "nombre": nombre})

# Esto redirige a la pagina del chatbot
@app.get("/chatbot", response_class=HTMLResponse)
async def chatbot_page(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request})

# Endpoint para recibir preguntas del chatbot
@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    pregunta = data.get("pregunta", "").lower()
    
    respuesta = responder_usuario(pregunta)
    return JSONResponse(content={"respuesta": respuesta})

# Ruta para mostrar información sobre café
@app.get("/info/cafe", response_class=HTMLResponse)
async def info_cafe(request: Request):
    return templates.TemplateResponse("informacion_cafe.html", {"request": request})

# Ruta para mostrar información sobre compostaje
@app.get("/info/compostaje", response_class=HTMLResponse)
async def info_compostaje(request: Request):
    return templates.TemplateResponse("informacion_compostaje.html", {"request": request})

import json
import numpy as np
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Cargar dataset JSON
with open('compostaje_dataset.json') as file:
    data = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ['?', '¿', '!', '.', ',']

# Tokenización y lematización
for intent in data['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

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

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compilar modelo
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenar
from tensorflow.keras.models import load_model
import pickle

model = load_model("chatbot_model.keras")
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import random


# Cargar el archivo
with open("compostaje_dataset.json", encoding='utf-8') as file:
    intents = json.load(file)

import re
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"[¿?¡!.,]", "", texto)  # Elimina los signos especificados
    return texto.strip()

def bag_of_words(sentence, words):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def responder_usuario(mensaje):
    mensaje = limpiar_texto(mensaje)
    bow = bag_of_words(mensaje, words)
    resultado = model.predict(np.array([bow]))[0]
    umbral = 0.7  # umbral de confianza
    resultados = [[i, r] for i, r in enumerate(resultado) if r > umbral]

    resultados.sort(key=lambda x: x[1], reverse=True)

    if resultados:
        tag = classes[resultados[0][0]]
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
    return "Lo siento, no entendí tu pregunta. ¿Puedes reformularla?"

# Ejemplo
entrada_usuario = "Hola"
respuesta = responder_usuario(entrada_usuario)
print(respuesta)
#prueba ramas

