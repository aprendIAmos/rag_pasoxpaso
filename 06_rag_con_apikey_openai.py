import os  # Permite leer variables de entorno.
from pathlib import Path  # Permite construir rutas a los archivos.
from dotenv import load_dotenv  # Carga las variables del archivo .env.
from openai import OpenAI  # Importa el cliente de OpenAI.
from pypdf import PdfReader  # Permite extraer texto del PDF.
import chromadb  # Importa la base de datos vectorial.


# RECOMENDACIONES: Instalá las bibliotecas de requirements.txt y colocá
# reglamento.pdf en esta carpeta. Copiá .env.example como .env y completá tu API key.
# Este paso usa OpenAI para los embeddings y la respuesta; no necesita Ollama.
# Para más instrucciones, revisá el archivo README.md de este proyecto.

carpeta = Path(__file__).resolve().parent  # Obtiene la carpeta del script.
load_dotenv(carpeta / ".env")  # Carga el .env de esa carpeta.

clave = os.getenv("OPENAI_API_KEY")  # Lee la clave de OpenAI desde el archivo .env.
modelo = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # Lee el modelo configurado, podés cambiarlo.
modelo_embeddings = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")  # Elige el modelo de vectores.

if not clave:  # Comprueba que exista una clave.
    raise SystemExit("Completá OPENAI_API_KEY en el archivo .env.")  # Detiene el programa si falta.

cliente = OpenAI(api_key=clave)  # Crea el cliente para generar embeddings y respuestas.

ruta_documento = carpeta / "reglamento.pdf"  # Construye la ruta al PDF.

if not ruta_documento.exists():  # Comprueba que el PDF exista.
    raise SystemExit("Colocá reglamento.pdf en la misma carpeta que este script.")  # Explica qué archivo falta.

lector = PdfReader(ruta_documento)  # Abre el PDF.
texto = ""  # Prepara una variable para reunir el contenido.

for pagina in lector.pages:  # Recorre las páginas del documento.
    texto_pagina = pagina.extract_text() or ""  # Extrae el texto de la página.
    texto += texto_pagina + "\n\n"  # Agrega el texto y un separador.

texto = " ".join(texto.split())  # Unifica espacios y saltos de línea.

if not texto:  # Comprueba que se haya extraído contenido.
    raise SystemExit("El PDF no tiene texto extraíble. Si es escaneado, necesita OCR.")  # Detiene el programa.

tamano_chunk = 800  # Define el tamaño máximo de cada fragmento en caracteres.
solapamiento = 150  # Define cuántos caracteres se repiten entre fragmentos.
paso = tamano_chunk - solapamiento  # Calcula cuánto avanzar en cada corte.

fragmentos = []  # Prepara la lista de chunks.
inicio = 0  # Indica desde qué carácter comienza el primer fragmento.

while inicio < len(texto):  # Continúa mientras quede texto.
    fin = inicio + tamano_chunk  # Calcula el final del fragmento.
    fragmento = texto[inicio:fin]  # Extrae el fragmento.
    fragmentos.append(fragmento)  # Lo agrega a la lista.
    if fin >= len(texto):  # Comprueba si se llegó al final del documento.
        break  # Termina para evitar repetir solamente el final.
    inicio += paso  # Avanza conservando los caracteres de solapamiento.

ids = [str(i) for i in range(len(fragmentos))]  # Genera un ID único por fragmento.

respuesta_embeddings = cliente.embeddings.create(  # Solicita un vector por fragmento a OpenAI.
    model=modelo_embeddings,  # Usa el modelo de embeddings configurado.
    input=fragmentos,  # Envía los textos del documento a la API.
)  # Termina la solicitud.
vectores = [dato.embedding for dato in sorted(respuesta_embeddings.data, key=lambda dato: dato.index)]  # Conserva el orden de los fragmentos.

base = chromadb.EphemeralClient()  # Crea una base temporal en memoria.
coleccion = base.create_collection(  # Crea la colección.
    name="biblioteca",  # Define su nombre.
    embedding_function=None,  # Los vectores los generamos nosotros con OpenAI.
)  # Termina la creación.

coleccion.add(  # Inserta los registros.
    ids=ids,  # Identifica cada fragmento.
    documents=fragmentos,  # Guarda los textos originales.
    embeddings=vectores,  # Guarda los vectores en el mismo orden que los textos.
)  # Termina la carga.

print("Páginas del PDF:", len(lector.pages))  # Muestra la cantidad de páginas.
print("Fragmentos guardados:", coleccion.count())  # Muestra la cantidad de chunks.

pregunta = input("Escribí tu pregunta: ").strip()  # Lee la pregunta del usuario.

if not pregunta:  # Comprueba que se haya escrito una pregunta.
    raise SystemExit("Escribí una pregunta para realizar la búsqueda.")  # Detiene el programa si está vacía.

respuesta_pregunta = cliente.embeddings.create(  # Convierte la pregunta en un vector.
    model=modelo_embeddings,  # Usa el mismo modelo que para los fragmentos.
    input=pregunta,  # Envía la pregunta a OpenAI.
)  # Termina la conversión.

resultado = coleccion.query(  # Busca los fragmentos más cercanos a la pregunta.
    query_embeddings=[respuesta_pregunta.data[0].embedding],  # Entrega el vector generado por OpenAI.
    n_results=min(3, len(fragmentos)),  # Recupera hasta tres fragmentos.
)  # Termina la búsqueda.

contexto = "\n\n".join(resultado["documents"][0])  # Une los fragmentos recuperados.

print("\nContexto recuperado:\n", contexto)  # Muestra el texto que recibirá OpenAI.

mensaje = f"""Contexto:
{contexto}

Pregunta:
{pregunta}"""  # Combina el contexto recuperado y la pregunta.

respuesta = cliente.chat.completions.create(  # Solicita una respuesta al modelo.
    model=modelo,  # Selecciona el modelo configurado.
    messages=[  # Prepara los mensajes.
        {  # Define las instrucciones.
            "role": "system",  # Identifica las instrucciones generales.
            "content": 'Respondé usando únicamente datos explícitos del contexto. Si falta el dato solicitado, respondé "No tengo esa información".',  # Establece cómo responder.
        },  # Cierra las instrucciones.
        {  # Define la consulta.
            "role": "user",  # Identifica el mensaje del usuario.
            "content": mensaje,  # Envía el contexto y la pregunta.
        },  # Cierra la consulta.
    ],  # Cierra la lista de mensajes.
)  # Termina la llamada.

print("\nRespuesta:", respuesta.choices[0].message.content)  # Muestra la respuesta.
