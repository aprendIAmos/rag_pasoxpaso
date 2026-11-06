import ollama  # Importa la biblioteca que genera los embeddings.
import chromadb  # Importa la base de datos vectorial.

# RECOMENDACIONES: Instalá las bibliotecas de requirements.txt y descargá
# embeddinggemma (embeddings) y qwen2.5:1.5b (respuestas) en Ollama.
# Para más instrucciones, revisá el archivo README.md de este proyecto.

textos = [  # Crea una lista con tres fragmentos del reglamento.
    "Los libros se pueden devolver de lunes a viernes de 9:00 a 18:30.",  # Fragmento sobre horarios.
    "El préstamo de libros dura 10 días.",  # Fragmento sobre duración.
    "Para obtener el carnet de socio hay que presentar el DNI.",  # Fragmento sobre inscripción.
]  # Cierra la lista de textos.

respuesta = ollama.embed(  # Solicita los vectores de los tres textos.
    model="embeddinggemma",  # Selecciona el modelo de embeddings.
    input=textos,  # Envía la lista completa de textos.
)  # Termina la llamada a Ollama.

vectores = respuesta.embeddings  # Obtiene la lista de vectores, uno por texto.

base = chromadb.EphemeralClient()  # Crea una base temporal en memoria.

coleccion = base.create_collection(  # Crea una colección para agrupar nuestros datos.
    name="biblioteca",  # Asigna un nombre a la colección.
    embedding_function=None,  # Indica que nosotros proporcionamos los vectores.
)  # Termina la creación de la colección.

coleccion.add(  # Agrega los tres registros a la colección.
    ids=["horarios", "prestamos", "socios"],  # Identifica cada registro con un ID único.
    documents=textos,  # Guarda los textos originales.
    embeddings=vectores,  # Guarda los vectores en el mismo orden que los textos.
)  # Termina la carga de datos.

pregunta = input("Escribí tu pregunta: ")  # Lee la consulta del usuario.

respuesta_pregunta = ollama.embed(  # Convierte la pregunta en un vector.
    model="embeddinggemma",  # Usa el mismo modelo que usamos para los textos.
    input=pregunta,  # Envía la pregunta.
)  # Termina la llamada a Ollama.

resultado = coleccion.query(  # Busca los vectores más cercanos al de la pregunta.
    query_embeddings=respuesta_pregunta.embeddings,  # Entrega la lista con el vector de la pregunta.
    n_results=1,  # Solicita solamente el primer resultado.
)  # Termina la búsqueda.

texto_recuperado = resultado["documents"][0][0]  # Obtiene el primer texto de la primera consulta.

print("Texto recuperado:", texto_recuperado)  # Muestra el fragmento encontrado.


respuesta_final = ollama.chat(  # Pide al modelo que redacte una respuesta.
    model="qwen2.5:1.5b",  # Usa el mismo modelo de chat de los primeros pasos.
    messages=[  # Prepara las instrucciones y la consulta.
        {  # Crea el mensaje de instrucciones.
            "role": "system",  # Indica que son instrucciones generales.
            "content": 'Respondé usando únicamente el contexto. Si falta el dato solicitado, decí "No tengo esa información".',  # Establece cómo responder.
        },  # Cierra el mensaje de instrucciones.
        {  # Crea el mensaje con el contexto y la pregunta.
            "role": "user",  # Identifica el mensaje del usuario.
            "content": f"Contexto:\n{texto_recuperado}\n\nPregunta:\n{pregunta}",  # Incorpora el fragmento recuperado y la consulta.
        },  # Cierra el mensaje del usuario.
    ],  # Cierra la lista de mensajes.
)  # Termina la llamada a Qwen.

print("Respuesta del modelo:", respuesta_final.message.content)  # Muestra la respuesta redactada.
