# -*- coding: utf-8 -*-
"""
Chatbot adaptado para español y formato Pregunta/Respuesta.
"""

import nltk
import numpy as np
import random
import string
import re  # Importamos la librería de expresiones regulares para procesar el archivo

# --- Descarga de recursos de NLTK (solo la primera vez) ---
# nltk.download('punkt')
# nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Lectura y Procesamiento del Archivo de Texto ---

# El archivo .txt tiene un formato específico: Pregunta -> Respuesta.
# Necesitamos leerlo y separar las preguntas de las respuestas en dos listas paralelas.
questions = []
answers = []

with open("chatbot.txt", "r", errors="ignore", encoding="utf-8") as f:
    raw_doc = f.read()

raw_doc = raw_doc.lower()  # Convertir todo a minúsculas

# Limpiamos los marcadores de fuente que no aportan información
raw_doc = re.sub(r"\', ", "", raw_doc)

# Dividimos el documento en bloques de pregunta y respuesta.
# Cada pregunta parece empezar con un número seguido de un punto.
qa_blocks = re.split(r"\n\d+\. ", raw_doc)

for block in qa_blocks:
    if "respuesta:" in block:
        # Dividimos el bloque en pregunta y respuesta
        parts = block.split("respuesta:")
        if len(parts) == 2:
            question = parts[0].strip()
            answer = parts[1].strip()
            questions.append(question)
            answers.append(answer)

# --- 2. Preprocesamiento de Texto en Español ---

# Usaremos un "stemmer" para español en lugar de un "lemmatizer" para inglés.
# Un stemmer reduce las palabras a su raíz (e.g., "jugadores" -> "jugador").
stemmer = nltk.stem.SnowballStemmer("spanish")
stop_words_es = nltk.corpus.stopwords.words(
    "spanish"
)  # Cargamos las "stop words" en español


def StemTokens(tokens):
    return [stemmer.stem(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


# LÍNEA CORRECTA
def StemNormalize(text):
    return StemTokens(
        nltk.word_tokenize(
            text.lower().translate(remove_punct_dict), language="spanish"
        )
    )


# --- 3. Definición de Saludos y Despedidas en Español ---

GREETING_INPUTS = ("hola", "buenas", "qué tal", "saludos", "hey", "buenos días")
GREETING_RESPONSES = [
    "¡Hola!",
    "¡Hey!",
    "¡Hola! ¿En qué puedo ayudarte?",
    "¡Buenas! Pregúntame algo sobre fútbol.",
]


def greeting(sentence):
    """Devuelve un saludo si la frase del usuario es un saludo."""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# --- 4. Generación de Respuestas ---


def response(user_response):
    """Genera una respuesta basada en la similitud del coseno."""
    robo_response = ""

    # Creamos un corpus temporal que incluye todas las preguntas + la pregunta del usuario
    temp_corpus = questions + [user_response]

    # Usamos el stemmer y las stop words en español
    TfidfVec = TfidfVectorizer(tokenizer=StemNormalize, stop_words=stop_words_es)
    tfidf = TfidfVec.fit_transform(temp_corpus)

    # Calculamos la similitud del coseno entre la pregunta del usuario (la última) y todas las preguntas del corpus
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])

    # Obtenemos el índice de la pregunta más similar
    idx = vals.argsort()[0][-1]

    # Obtenemos el valor de similitud
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-1]

    if req_tfidf == 0:
        robo_response = (
            "Lo siento, no te he entendido. ¿Puedes preguntarme de otra forma?"
        )
    else:
        # Devolvemos la RESPUESTA correspondiente a la PREGUNTA más similar
        robo_response = answers[idx]

    return robo_response


# --- 5. Bucle Principal del Chatbot ---

flag = True
print(
    "ROBO: ¡Hola! Soy un bot experto en fútbol. Pregúntame lo que quieras. Para salir, escribe 'adiós'."
)

while flag:
    user_response = input("> ")
    user_response = user_response.lower()

    if user_response not in ["adiós", "bye", "chao", "salir"]:
        if user_response in ["gracias", "muchas gracias"]:
            flag = False
            print("ROBO: De nada. ¡Ha sido un placer!")
        else:
            if greeting(user_response) is not None:
                print("ROBO: " + greeting(user_response))
            else:
                print("ROBO: ", end="")
                print(response(user_response))
    else:
        flag = False
        print("ROBO: ¡Adiós! ¡Que tengas un buen día!")
