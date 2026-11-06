import os  # Permite leer las variables de entorno.
from pathlib import Path  # Permite construir rutas a los archivos.
from uuid import uuid4  # Permite generar nombres únicos para las colecciones.
from dotenv import load_dotenv  # Carga la configuración del archivo .env.
from openai import OpenAI  # Permite consultar el modelo de OpenAI.
from pypdf import PdfReader  # Permite extraer texto del PDF.
import chromadb  # Proporciona la base de datos vectorial.
import streamlit as st  # Permite crear la interfaz web.

# Instalá requirements.txt, colocá reglamento.pdf en esta carpeta y configurá tu API key.
# Usá .env localmente o Secrets en Streamlit Community Cloud; no hace falta Ollama.
# Los embeddings y las respuestas se generan con OpenAI.
# Para más instrucciones, revisá el archivo README.md de este proyecto.

st.set_page_config(page_title="Chat con el reglamento", page_icon="📚")  # Configura la página.

carpeta = Path(__file__).resolve().parent  # Obtiene la carpeta de esta aplicación.
load_dotenv(carpeta / ".env")  # Carga las variables del .env.

clave = os.getenv("OPENAI_API_KEY")  # Lee la clave de OpenAI.
modelo = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # Lee el modelo configurado.
modelo_embeddings = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")  # Elige el modelo de vectores.
ruta_pdf = carpeta / "reglamento.pdf"  # Localiza el PDF que vamos a consultar.

st.title("📚 Chat con el reglamento")  # Muestra el título de la aplicación.
st.caption("Preguntá sobre el contenido de reglamento.pdf.")  # Explica qué documento se consulta.

if not clave:  # Comprueba que exista una clave.
    st.error("Completá OPENAI_API_KEY en .env o en Secrets de Streamlit Community Cloud.")  # Muestra qué configuración falta.
    st.stop()  # Detiene esta ejecución de la aplicación.

if not ruta_pdf.exists():  # Comprueba que el documento esté disponible.
    st.error("Colocá reglamento.pdf en la misma carpeta que este script.")  # Explica qué archivo falta.
    st.stop()  # Detiene esta ejecución.

cliente = OpenAI(api_key=clave)  # Crea el cliente para generar embeddings y respuestas.

if "coleccion" not in st.session_state:  # Procesa el PDF solo una vez por sesión.
    with st.spinner("Preparando el documento y la base vectorial..."):  # Muestra un aviso mientras trabaja.
        lector = PdfReader(ruta_pdf)  # Abre el PDF.
        texto = ""  # Prepara una variable para reunir el contenido.

        for pagina in lector.pages:  # Recorre todas las páginas.
            texto_pagina = pagina.extract_text() or ""  # Extrae el texto disponible.
            texto += texto_pagina + "\n\n"  # Agrega el contenido de la página.

        texto = " ".join(texto.split())  # Unifica espacios y saltos de línea.

        if not texto:  # Comprueba que se haya extraído texto.
            st.error("El PDF no contiene texto extraíble. Si es escaneado, necesita OCR.")  # Informa el problema.
            st.stop()  # Detiene esta ejecución.

        tamano_chunk = 800  # Define el tamaño máximo de cada fragmento en caracteres.
        solapamiento = 150  # Define cuántos caracteres comparten los fragmentos consecutivos.
        paso = tamano_chunk - solapamiento  # Calcula cuánto avanzar entre cortes.

        fragmentos = []  # Prepara la lista de fragmentos.
        inicio = 0  # Indica dónde comienza el primer corte.

        while inicio < len(texto):  # Continúa mientras quede contenido.
            fin = inicio + tamano_chunk  # Calcula el final del fragmento.
            fragmentos.append(texto[inicio:fin])  # Guarda el fragmento.
            if fin >= len(texto):  # Comprueba si ya se llegó al final.
                break  # Evita generar otro fragmento que solo repita el final.
            inicio += paso  # Avanza conservando el solapamiento.

        try:  # Evita mostrar detalles internos si falla la API al preparar el documento.
            respuesta_embeddings = cliente.embeddings.create(  # Solicita un vector por fragmento.
                model=modelo_embeddings,  # Usa el modelo de embeddings configurado.
                input=fragmentos,  # Envía los textos del documento a OpenAI.
            )  # Termina la solicitud.
        except Exception:  # Captura problemas de conexión, credenciales o cuota.
            st.error("No se pudieron generar los embeddings. Revisá la conexión, la API key, el modelo y la cuota de OpenAI.")  # Informa el fallo.
            st.stop()  # Permite reintentar sin guardar una colección incompleta.

        vectores = [dato.embedding for dato in sorted(respuesta_embeddings.data, key=lambda dato: dato.index)]  # Conserva el orden de los fragmentos.

        base = chromadb.EphemeralClient()  # Crea un cliente de base temporal en memoria.
        coleccion = base.create_collection(  # Crea una colección para esta sesión.
            name="biblioteca_" + uuid4().hex,  # Genera un nombre único para evitar conflictos entre sesiones.
            embedding_function=None,  # OpenAI genera los vectores; Chroma solo los almacena y busca.
        )  # Termina la creación.

        coleccion.add(  # Agrega los textos y los vectores ya generados.
            ids=[str(i) for i in range(len(fragmentos))],  # Asigna un ID único a cada fragmento.
            documents=fragmentos,  # Entrega los textos a ChromaDB.
            embeddings=vectores,  # Entrega los vectores en el mismo orden.
        )  # Termina la carga.

        st.session_state["base"] = base  # Conserva el cliente durante la sesión.
        st.session_state["coleccion"] = coleccion  # Conserva la colección para las próximas preguntas.
        st.session_state["modelo_embeddings"] = modelo_embeddings  # Recuerda el modelo usado al indexar.

if "mensajes" not in st.session_state:  # Comprueba si ya existe el historial visual.
    st.session_state["mensajes"] = []  # Crea una lista para los mensajes.

for mensaje in st.session_state["mensajes"]:  # Recorre las conversaciones guardadas.
    with st.chat_message(mensaje["role"]):  # Dibuja el mensaje como usuario o asistente.
        st.markdown(mensaje["content"])  # Muestra el contenido.
        if "contexto" in mensaje:  # Comprueba si el mensaje tiene fragmentos asociados.
            with st.expander("Ver contexto recuperado"):  # Ofrece mostrar la información utilizada.
                st.write(mensaje["contexto"])  # Muestra los fragmentos enviados al modelo.

pregunta = st.chat_input("Escribí tu pregunta sobre el reglamento")  # Muestra la entrada del chat.

if pregunta and pregunta.strip():  # Procesa la pregunta cuando tiene contenido.
    pregunta = pregunta.strip()  # Elimina espacios en los extremos.

    st.session_state["mensajes"].append(  # Guarda la pregunta en el historial visual.
        {"role": "user", "content": pregunta}  # Define el mensaje del usuario.
    )  # Termina el guardado.

    with st.chat_message("user"):  # Dibuja el nuevo mensaje del usuario.
        st.markdown(pregunta)  # Muestra su pregunta.

    with st.chat_message("assistant"):  # Prepara el espacio para la respuesta.
        try:  # Permite mostrar un aviso si la búsqueda o la consulta fallan.
            with st.spinner("Buscando información y preparando la respuesta..."):  # Muestra el progreso.
                coleccion = st.session_state["coleccion"]  # Recupera la colección ya preparada.

                respuesta_pregunta = cliente.embeddings.create(  # Convierte la pregunta en un vector.
                    model=st.session_state["modelo_embeddings"],  # Usa el mismo modelo que para el documento.
                    input=pregunta,  # Envía la pregunta a OpenAI.
                )  # Termina la conversión.

                resultado = coleccion.query(  # Busca los fragmentos relacionados con la pregunta.
                    query_embeddings=[respuesta_pregunta.data[0].embedding],  # Entrega el vector de la consulta.
                    n_results=min(3, coleccion.count()),  # Recupera hasta tres fragmentos.
                )  # Termina la búsqueda.

                contexto = "\n\n".join(resultado["documents"][0])  # Une los fragmentos recuperados.

                mensaje_modelo = f"""Contexto:
{contexto}

Pregunta:
{pregunta}"""  # Combina la información encontrada y la pregunta actual.

                respuesta = cliente.chat.completions.create(  # Envía la consulta al modelo.
                    model=modelo,  # Selecciona el modelo configurado.
                    messages=[  # Define los mensajes que recibirá el modelo.
                        {  # Define las instrucciones.
                            "role": "system",  # Identifica las reglas generales.
                            "content": 'Respondé usando únicamente datos explícitos del contexto. Si falta el dato solicitado, respondé "No tengo esa información".',  # Establece cómo responder.
                        },  # Cierra las instrucciones.
                        {  # Define la consulta.
                            "role": "user",  # Identifica el mensaje del usuario.
                            "content": mensaje_modelo,  # Envía el contexto y la pregunta.
                        },  # Cierra la consulta.
                    ],  # Cierra la lista de mensajes.
                )  # Termina la llamada a OpenAI.

                texto_respuesta = respuesta.choices[0].message.content or "El modelo devolvió una respuesta vacía."  # Obtiene el texto.

            st.markdown(texto_respuesta)  # Muestra la respuesta del asistente.

            with st.expander("Ver contexto recuperado"):  # Permite inspeccionar los fragmentos utilizados.
                st.write(contexto)  # Muestra el contexto.

            st.session_state["mensajes"].append(  # Conserva la respuesta y su contexto.
                {"role": "assistant", "content": texto_respuesta, "contexto": contexto}  # Define el mensaje del asistente.
            )  # Termina el guardado.

        except Exception:  # Captura un fallo durante la consulta.
            st.error("No se pudo completar la consulta. Revisá la conexión y la configuración de OpenAI.")  # Informa el fallo.
