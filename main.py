#Se inpottan las libresias necesarias
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

#Se crea la app fastapi
app = FastAPI()

# Montar la carpeta 'static' para servir archivos estáticos como CSS e imágenes
app.mount("/static", StaticFiles(directory="static"), name="static")

# Se define el directorio donde están las plantillas HTML (usando Jinja2
templates = Jinja2Templates(directory="templates")

# Ruta principal ("/") que muestra la página de inicio con un formulario para ingresar nombre
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Ruta del menú principal. Recibe el nombre del usuario como parámetro y lo muestra en pantalla.
@app.get("/menu", response_class=HTMLResponse)
async def menu(request: Request, nombre: str = ""):
    return templates.TemplateResponse("menu.html", {"request": request, "nombre": nombre})

# Ruta que lleva a la página del chatbot, donde se puede conversar con la IA.
@app.get("/chatbot", response_class=HTMLResponse)
async def chatbot_page(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request})

# Este endpoint recibe las preguntas enviadas desde la interfaz del chatbot (vía JavaScript),
# procesa la pregunta con la IA, y devuelve una respuesta en formato JSON.
@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    palabras = ["Compostaje", "Café"]
    valida = 0
    '''if palabras in data:
        pregunta = data.get("pregunta", "").lower()
        respuesta = responder_usuario(pregunta)
        return JSONResponse(content={"respuesta": "¡Hola! ¿En qué puedo ayudarte con el compostaje o el café?"})  
    else:'''
    pregunta = data.get("pregunta", "").lower()
    respuesta = responder_usuario(pregunta)
    return JSONResponse(content={"respuesta": respuesta})

# Página que contiene información educativa sobre el café.
@app.get("/info/cafe", response_class=HTMLResponse)
async def info_cafe(request: Request):
    return templates.TemplateResponse("informacion_cafe.html", {"request": request})

# Página con información educativa sobre compostaje (abonos orgánicos).
@app.get("/info/compostaje", response_class=HTMLResponse)
async def info_compostaje(request: Request):
    return templates.TemplateResponse("informacion_compostaje.html", {"request": request})

# Se importan librerías para trabajar con texto, crear modelos de redes neuronales y optimizarlos.
import json
import numpy as np
import random
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import keras as keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
nltk.download('punkt_tab')


# Verifica que los recursos de NLTK estén disponibles. Si no lo están, los descarga automáticamente.
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
    
# Inicializa el lematizador para reducir palabras a su forma base (por ejemplo, "plantando" → "plantar").
lemmatizer = WordNetLemmatizer()

# Carga el archivo JSON con los "intents", que contiene patrones de entrada (preguntas del usuario)
# Procesamiento de datos: tokenización, lematización, eliminación de símbolos
with open('compostaje_dataset.json') as file:
    data = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ['?', '¿', '!', '.', ',']

for intent in data['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lematizar, eliminar duplicados y ordenar palabras y clases
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Crear dataset de entrenamiento (bag of words)
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

# Mezclar datos y convertir a numpy arrays
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Crear modelo de red neuronal (ya entrenado, pero está definido aquí)
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compilar modelo con optimizador SGD
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) # type: ignore

#Cargar modelo entrenado 
from keras.models import load_model
import pickle

# Cargar modelo previamente entrenado y serializado
model = load_model("chatbot_model.keras")
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Desactivar optimizaciones de OneDNN para compatibilidad (opcional)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Cargar el archivo
with open("compostaje_dataset.json", encoding='utf-8') as file:
    intents = json.load(file)

#Logica del chatbot
import re

# Función para limpiar texto del usuario
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"[¿?¡!.,]", "", texto)  # Elimina los signos especificados
    return texto.strip()

# Convertir frase a bag of words
def bag_of_words(sentence, words):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

# Responder al usuario basado en la predicción del modelo
def responder_usuario(mensaje):
    mensaje = limpiar_texto(mensaje)
    bow = bag_of_words(mensaje, words)
    resultado = model.predict(np.array([bow]))[0] # type: ignore
    umbral = 0.7  # umbral de confianza
    # Filtrar respuestas con confianza suficiente
    resultados = [[i, r] for i, r in enumerate(resultado) if r > umbral]
    resultados.sort(key=lambda x: x[1], reverse=True)

    # Si hay respuesta válida, buscar la correspondiente
    if resultados:
        tag = classes[resultados[0][0]]
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
    # Si no se entiende la pregunta
    return "Lo siento, no entendí tu pregunta. ¿Puedes reformularla?"

# Ejemplo
entrada_usuario = "Hola"
respuesta = responder_usuario(entrada_usuario)
print(respuesta)
#prueba ramas


