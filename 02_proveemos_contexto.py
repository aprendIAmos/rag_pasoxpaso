import ollama  # Importa la biblioteca para comunicarnos con Ollama.

# RECOMENDACIONES: No te olvides de instalar la biblioteca de Ollama y de 
# descargar el modelo qwen2.5:1.5b antes de ejecutar este script.
# Para más instrucciones, revisá el archivo README.md de este proyecto.


contexto = "En la Biblioteca del Sur, los libros se pueden devolver de lunes a viernes de 9:00 a 18:30."  # Define el reglamento.

pregunta = input("Escribí tu pregunta: ")  # Guarda la pregunta del usuario.
# Pregunta1: ¿Hasta qué hora puedo devolver un libro en la Biblioteca del Sur?
# Preguenta2: ¿Cuántos días dura el préstamo?

#Respondé la pregunta con información explícita del contexto.
# No supongas ni inventes datos.
# Si el contexto no permite responder, escribí exactamente: "No tengo esa información".
mensaje = f"""Respondé usando únicamente el contexto. Si la respuesta no aparece, decí que no tenés esa información.

Contexto:
{contexto}

Pregunta:
{pregunta}"""  # Combina la instrucción, el contexto y la pregunta.

respuesta = ollama.chat(  # Envía la consulta al modelo.
    model="qwen2.5:1.5b",  # Selecciona el modelo descargado.
    messages=[  # Crea la lista de mensajes.
        {"role": "user", "content": mensaje}  # Envía el texto completo.
    ],  # Cierra la lista de mensajes.
)  # Termina la llamada.

print(respuesta.message.content)  # Muestra la respuesta.
