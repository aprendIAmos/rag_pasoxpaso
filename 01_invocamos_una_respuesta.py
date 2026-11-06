import ollama  # Importa la biblioteca para comunicarnos con Ollama.

# RECOMENDACIONES: No te olvides de instalar la biblioteca de Ollama y de 
# descargar el modelo qwen2.5:1.5b antes de ejecutar este script.
# Para más instrucciones, revisá el archivo README.md de este proyecto.



# ¿Hasta qué hora puedo devolver un libro en la Biblioteca del Sur?
pregunta = input("Ingrese su pregunta: ")

respuesta = ollama.chat(  # Envía una consulta y espera la respuesta completa.
    model="qwen2.5:1.5b",  # Elige el modelo que descargamos.
    messages=[  # Crea la lista de mensajes que recibirá el modelo.
        {"role": "user", "content": pregunta}  # Incluye la pregunta.
    ],  # Cierra la lista de mensajes.
)  # Termina la llamada a Ollama.

print(respuesta.message.content)  # Muestra únicamente el texto de la respuesta.
