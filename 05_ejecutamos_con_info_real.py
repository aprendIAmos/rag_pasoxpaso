from pathlib import Path  # Permite localizar el PDF junto al script.
from pypdf import PdfReader  # Permite extraer texto de un PDF.
import ollama  # Genera embeddings y respuestas con modelos locales.
import chromadb  # Guarda los fragmentos y sus vectores.

# RECOMENDACIONES: Instalá las bibliotecas de requirements.txt y descargá
# embeddinggemma (embeddings) y qwen2.5:1.5b (respuestas) en Ollama.
# Colocá reglamento.pdf en la misma carpeta que este script.
# Para más instrucciones, revisá el archivo README.md de este proyecto.

ruta_pdf = Path(__file__).with_name("reglamento.pdf")  # Construye la ruta al PDF.
lector = PdfReader(ruta_pdf)  # Abre el documento.

texto_completo = ""  # Prepara una variable para reunir el texto.

for pagina in lector.pages:  # Recorre todas las páginas del PDF.
    texto_pagina = pagina.extract_text() or ""  # Extrae el texto o usa una cadena vacía.
    texto_completo += texto_pagina + "\n\n"  # Agrega la página y un separador.

texto_completo = " ".join(texto_completo.split())  # Unifica espacios y saltos de línea.

if not texto_completo:  # Comprueba que se haya extraído contenido.
    raise SystemExit("No se encontró texto. Si el PDF es escaneado, necesita OCR.")  # Detiene el programa.

tamano_chunk = 800  # Define la cantidad máxima de caracteres de cada fragmento.
solapamiento = 150  # Define cuántos caracteres se repiten en el siguiente fragmento.
paso = tamano_chunk - solapamiento  # Calcula cuánto avanzar al crear cada fragmento.

textos = []  # Prepara la lista que recibirá los fragmentos.
inicio = 0  # Indica desde qué carácter empezamos a cortar.

while inicio < len(texto_completo):  # Continúa mientras quede texto.
    fin = inicio + tamano_chunk  # Calcula el final del fragmento.
    fragmento = texto_completo[inicio:fin]  # Extrae hasta 800 caracteres.
    textos.append(fragmento)  # Guarda el fragmento.
    if fin >= len(texto_completo):  # Comprueba si ya se llegó al final del documento.
        break  # Evita crear otro fragmento que solo repita el final.
    inicio += paso  # Avanza conservando el solapamiento.

print("Cantidad de páginas:", len(lector.pages))  # Muestra cuántas páginas tiene el PDF.
print("Cantidad de fragmentos:", len(textos))  # Muestra cuántos chunks se crearon.
print("Primer fragmento:", textos[0])  # Permite revisar el primer chunk.

respuesta = ollama.embed(  # Solicita un vector por fragmento.
    model="embeddinggemma",  # Selecciona el modelo de embeddings.
    input=textos,  # Envía los fragmentos extraídos del PDF.
)  # Termina la solicitud.

vectores = respuesta.embeddings  # Obtiene los vectores de los fragmentos.

base = chromadb.EphemeralClient()  # Crea una base temporal en memoria.

coleccion = base.create_collection(  # Crea la colección del documento.
    name="biblioteca",  # Define el nombre de la colección.
    embedding_function=None,  # Indica que proporcionamos los embeddings.
)  # Termina la creación.

ids = [str(i) for i in range(len(textos))]  # Genera un ID único para cada fragmento.

coleccion.add(  # Guarda los registros.
    ids=ids,  # Entrega los identificadores.
    documents=textos,  # Guarda los fragmentos originales.
    embeddings=vectores,  # Guarda sus vectores en el mismo orden.
)  # Termina la carga.

pregunta = input("Escribí tu pregunta: ")  # Lee la pregunta del usuario.

respuesta_pregunta = ollama.embed(  # Convierte la pregunta en un vector.
    model="embeddinggemma",  # Usa el mismo modelo que para los fragmentos.
    input=pregunta,  # Envía la pregunta.
)  # Termina la conversión.

resultado = coleccion.query(  # Busca los fragmentos más relacionados.
    query_embeddings=respuesta_pregunta.embeddings,  # Entrega el vector de la pregunta.
    n_results=min(3, len(textos)),  # Recupera hasta tres fragmentos.
)  # Termina la búsqueda.

contexto = "\n\n".join(resultado["documents"][0])  # Une los fragmentos de la primera consulta.

print("\nContexto recuperado:\n", contexto)  # Muestra qué información recibirá Qwen.

respuesta_final = ollama.chat(  # Solicita una respuesta basada en los fragmentos.
    model="qwen2.5:1.5b",  # Usa el modelo de chat que veníamos utilizando.
    messages=[  # Construye los mensajes.
        {  # Define las instrucciones.
            "role": "system",  # Identifica el mensaje de instrucciones.
            "content": 'Respondé usando únicamente el contexto. Si falta el dato solicitado, decí "No tengo esa información".',  # Establece cómo responder.
        },  # Cierra las instrucciones.
        {  # Define la consulta.
            "role": "user",  # Identifica el mensaje del usuario.
            "content": f"Contexto:\n{contexto}\n\nPregunta:\n{pregunta}",  # Incluye los fragmentos y la pregunta.
        },  # Cierra la consulta.
    ],  # Cierra la lista de mensajes.
)  # Termina la llamada.

print("\nRespuesta del modelo:", respuesta_final.message.content)  # Muestra la respuesta.
