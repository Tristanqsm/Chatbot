# CHATBOT1
from nltk.chat.util import Chat, reflections

pares = [
    (
        r"mi nombre es (.*)",
        [
            "Hola %1, como estas?",
        ],
    ),
    (r"cual es tu nombre?", ["Mi nombre es Chatbot ?"]),
    (r"como estas ?", ["Bien, y tu?"]),
    (r"disculpa (.*)", ["No pasa nada"]),
    (r"hola|hey|buenas", ["Hola", "Que tal"]),
    (r"que (.*) quieres", ["Nada gracias"]),
    (r"(.*) creado ?", ["Fui creado hoy"]),
]
chat = Chat(pares, reflections)
print("Hola, soy Chatbot y estoy aqui para ayudarte")
chat.converse()
