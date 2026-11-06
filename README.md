# RAG paso a paso

Una introducción práctica a **RAG (Retrieval-Augmented Generation)**, o generación aumentada por recuperación, usando Python.

Vamos a pasar de hacerle una pregunta a un modelo a crear una aplicación web que responde preguntas sobre un PDF.

## ¿Qué es RAG?

En lugar de esperar que el modelo conozca nuestro documento, buscamos fragmentos relevantes y se los entregamos como contexto para responder.

```text
Preparación: documento → fragmentos → embeddings → base vectorial
Consulta: pregunta → embedding → búsqueda → contexto + pregunta → modelo → respuesta
```

- **Embedding:** representación numérica de un texto que permite comparar similitud.
- **Base vectorial:** almacena esos vectores y permite buscar fragmentos relacionados con una pregunta. Aquí usamos ChromaDB.
- **Chunk o fragmento:** una parte del documento. Dividirlo permite recuperar solo la información que necesitamos.
- **Solapamiento:** texto que se repite entre fragmentos consecutivos para conservar parte del contexto del corte.

**RAG no entrena al modelo ni modifica sus conocimientos.** Le proporciona información al momento de responder. Tampoco garantiza respuestas correctas: conviene revisar los fragmentos recuperados.

## 1. Requisitos

Esta guía usa **Windows y PowerShell**. Ejecutá los comandos desde la carpeta del proyecto, salvo que se indique otra cosa.

Necesitás:

- Python instalado. Python 3.11 es una opción conservadora para este proyecto; podés descargarlo desde <https://www.python.org/downloads/>.
- Conexión a Internet para instalar dependencias y descargar modelos.
- Espacio en disco para los modelos y memoria disponible para ejecutarlos. El rendimiento depende de tu equipo.
- Una cuenta de OpenAI con acceso a la API para los pasos 06 y 07. Los pasos 01 a 05 no requieren una API key.

Descargá el proyecto desde GitHub con **Code → Download ZIP**, descomprimilo y abrí una terminal en su carpeta. Si tenés Git y la dirección del repositorio, también podés clonarlo.

Verificá Python:

```powershell
py --version
```

Si `py` no está disponible, probá `python --version` y reemplazá `py` por `python` en la creación del entorno.

## 2. Instalar Ollama y descargar los modelos

[Ollama](https://ollama.com/) permite ejecutar modelos en tu computadora. En PowerShell, el comando de instalación oficial para Windows es:

```powershell
irm https://ollama.com/install.ps1 | iex
```

Este comando descarga y ejecuta un script del sitio oficial. Si preferís revisar el instalador o descargarlo manualmente, visitá <https://ollama.com/download/windows>.

Completá la instalación. Si el comando `ollama` no se reconoce, cerrá la terminal y abrí una nueva.

```powershell
ollama --version
ollama pull qwen2.5:1.5b
ollama pull embeddinggemma
ollama list
```

- `qwen2.5:1.5b` genera las respuestas de los pasos 01, 02, 04 y 05.
- `embeddinggemma` convierte textos y preguntas en vectores en los pasos 03 a 05. **No es el modelo que redacta las respuestas.**

Todos los ejemplos de generación local usan el mismo modelo para simplificar la instalación. El paso 03 solo recupera texto: utiliza `embeddinggemma`, pero no Qwen.

Ollama debe estar en ejecución para los pasos locales. En Windows normalmente queda activo en segundo plano después de instalarlo.

> Instalar la biblioteca Python `ollama` no instala la aplicación Ollama ni descarga los modelos: son pasos diferentes.

## 3. Crear el entorno virtual e instalar las bibliotecas

Desde la carpeta del proyecto:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

El entorno virtual mantiene las bibliotecas del proyecto separadas de las de otros proyectos. Si recibiste una carpeta `venv` junto con el código, no la reutilices: los entornos deben crearse en cada equipo.

`requirements.txt` incluye todas las bibliotecas necesarias: `ollama`, `chromadb`, `pypdf`, `python-dotenv`, `openai` y `streamlit`.

### Si PowerShell bloquea la activación

No es necesario cambiar la política de seguridad: podés usar directamente el Python del entorno.

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

En ese caso, reemplazá `python` por `.\venv\Scripts\python.exe` en los comandos siguientes.

Si usás la activación, recordá repetirla cada vez que abras una terminal nueva. Para salir del entorno:

```powershell
deactivate
```

## 4. Recorrer los ejemplos

Cada script es independiente. Ejecutalos en orden para entender qué agrega cada uno.

### Paso 01: preguntarle al modelo sin contexto

```powershell
python 01_invocamos_una_respuesta.py
```

Probá: **¿Hasta qué hora puedo devolver un libro en la Biblioteca del Sur?**

El modelo no recibió el reglamento. Observá si reconoce que no sabe o si propone una respuesta sin fundamento. Una respuesta convincente no necesariamente es verdadera.

### Paso 02: proporcionar contexto manualmente

```powershell
python 02_proveemos_contexto.py
```

Ahora el código incluye una oración con el horario de devolución. Probá la misma pregunta y compará con el paso anterior.

Volvé a ejecutar el script y preguntá: **¿Cuántos días dura el préstamo?** Ese dato no está en el contexto de este paso: lo esperado es que el modelo indique que no tiene la información.

Todavía no hay recuperación automática: nosotros elegimos el contexto.

### Paso 03: generar embeddings y recuperar información

```powershell
python 03_embebemos_el_contexto.py
```

El script transforma tres textos en vectores, los guarda en ChromaDB y busca el más parecido a la pregunta.

Probá: **¿Qué necesito para hacerme socio?**

Observá el texto recuperado. En este paso **no se genera una respuesta final**: estamos aprendiendo la parte de recuperación.

### Paso 04: completar el circuito RAG

```powershell
python 04_inyectamos_el_texto_recuperado.py
```

Se recupera un fragmento y se lo entrega al modelo junto con la pregunta. Probá: **¿Cuántos días puedo tener prestado un libro?**

Compará el texto recuperado con la respuesta redactada. Aquí ya tenemos un RAG básico completo.

Este script usa `qwen2.5:1.5b`, igual que los primeros pasos.

### Paso 05: consultar un PDF real

```powershell
python 05_ejecutamos_con_info_real.py
```

El archivo `reglamento.pdf` debe estar en la misma carpeta que el script.

El programa extrae su texto, lo divide en fragmentos de hasta **800 caracteres**, con **150 caracteres de solapamiento**, y recupera hasta tres fragmentos para responder.

Esas cantidades son caracteres, no tokens. Son valores didácticos, no una configuración universal para cualquier documento.

Elegí una pregunta cuya respuesta hayas leído en el PDF. Revisá la cantidad de fragmentos, el contexto recuperado y la respuesta. Después probá una pregunta que el documento no pueda responder.

Si el PDF es una imagen escaneada sin texto seleccionable, necesitarás aplicar OCR antes: `pypdf` no hace reconocimiento de imágenes.

### Paso 06: generar embeddings y respuestas con OpenAI

Primero configurá la API key siguiendo la sección siguiente. Después ejecutá:

```powershell
python 06_rag_con_apikey_openai.py
```

Este paso mantiene el circuito que aprendimos con Ollama, pero usa OpenAI para las dos tareas: **`text-embedding-3-small` genera los embeddings** y **`gpt-4.1-mini` redacta la respuesta**, salvo que configures otros modelos.

El código llama explícitamente a `cliente.embeddings.create()` para los fragmentos y para la pregunta. Ambos deben usar el mismo modelo de embeddings. Chroma recibe los vectores mediante `embeddings` y busca mediante `query_embeddings`; su colección tiene `embedding_function=None`, por lo que no genera vectores automáticamente.

Este script no necesita Ollama ni Qwen y no descarga un modelo de embeddings de Chroma. Necesita conexión a OpenAI y una API key válida.

**Privacidad y costos:** todos los fragmentos del PDF se envían a OpenAI para generar sus embeddings. También se envía la pregunta para convertirla en un vector y, después, la pregunta con el contexto recuperado para redactar la respuesta. Ambas tareas consumen API. Cada ejecución del paso 06 vuelve a generar los embeddings del documento.

### Paso 07: agregar una interfaz con Streamlit

```powershell
python -m streamlit run 07_rag_con_interfaz_streamlit.py
```

Abrí la dirección que aparece en la terminal, normalmente <http://localhost:8501>. Para detener la aplicación, presioná `Ctrl+C` en esa terminal.

La aplicación permite hacer preguntas y desplegar el contexto recuperado. Usa OpenAI para los embeddings y las respuestas, como el paso 06; Chroma solo almacena y busca los vectores. No necesita Ollama.

El PDF y sus embeddings se preparan una vez por sesión, no con cada pregunta. Cada nueva pregunta sí requiere su propio embedding y una llamada de generación. Una sesión nueva vuelve a generar los embeddings del PDF y genera consumo adicional. Si cambiás el modelo de embeddings, reiniciá la aplicación para reconstruir la colección.

El historial que ves es **visual**: las preguntas anteriores no se envían al modelo. Para esta versión, escribí preguntas completas en lugar de referencias como “¿y cuánto dura eso?”.

## 5. Crear el archivo .env y obtener una API key

Esta configuración se necesita solamente para los pasos 06 y 07.

### Obtener una clave de OpenAI

1. Ingresá a <https://platform.openai.com/> y creá una cuenta o iniciá sesión.
2. Seleccioná o creá un proyecto.
3. Revisá la sección de facturación y configurá un medio de pago o saldo si corresponde.
4. Entrá a **API keys**: <https://platform.openai.com/api-keys>.
5. Creá una nueva clave y guardala en un lugar seguro. No la compartas ni la pegues en capturas o chats.

Los nombres de las opciones pueden variar con el tiempo. **La API tiene facturación propia: una suscripción a ChatGPT no incluye automáticamente su consumo.** Revisá precios, uso y alertas de presupuesto en tu cuenta.

### Consultar modelos y precios

Consultá las tarifas en la [página oficial de precios de OpenAI](https://developers.openai.com/api/docs/pricing). Buscá los modelos que usamos en este proyecto:

- **`gpt-4.1-mini`:** generación de respuestas. Revisá las tarifas de tokens de entrada (instrucciones, contexto y pregunta) y de salida (respuesta generada).
- **`text-embedding-3-small`:** generación de embeddings. Revisá la tarifa por los tokens del texto procesado, tanto del documento como de las preguntas.

Las tarifas se expresan por cantidad de tokens, no por cantidad de preguntas. El costo depende del modelo, del texto procesado y de la extensión de las respuestas. Recordá que cada ejecución del paso 06 y cada sesión nueva del paso 07 vuelven a generar los embeddings del PDF.

**Los precios pueden cambiar:** consultá siempre la página oficial. Para ver el consumo real de tu cuenta, visitá <https://platform.openai.com/usage>.

### Copiar la plantilla

El archivo `.env.example` es una plantilla sin credenciales. Hacé una copia en la misma carpeta y llamala exactamente `.env`, sin agregar `.txt`.

Podés hacerlo desde el explorador o con PowerShell:

```powershell
Copy-Item .env.example .env
```

**Si ya existe `.env`, no ejecutes la copia: abrí el que tenés para no sobrescribir tu configuración.**

Editá `.env` y completalo así, reemplazando el texto de ejemplo por tu clave real:

```dotenv
OPENAI_API_KEY=tu_clave_real
OPENAI_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

`OPENAI_MODEL` es opcional: ese es el modelo predeterminado del código. Podés elegir otro modelo disponible en tu cuenta que sea compatible con la llamada de chat utilizada.

`OPENAI_EMBEDDING_MODEL` también es opcional: su valor predeterminado es `text-embedding-3-small`. Debe ser un modelo de embeddings, no uno de chat. Se usa tanto para indexar el documento como para convertir las preguntas.

No hace falta copiar la clave dentro de los scripts: `python-dotenv` carga el archivo y el programa lee las variables de entorno. Si cambiás la configuración con la aplicación abierta, reiniciala.

> **Seguridad y privacidad:** nunca subas `.env` a GitHub. El `.gitignore` del proyecto ya lo excluye. En los pasos 06 y 07, todos los fragmentos del PDF y las preguntas se envían a OpenAI para generar embeddings; además se envían la pregunta y el contexto recuperado para responder. Usá documentos cuyo contenido tengas permiso de procesar con ese servicio.

## 6. Publicar en Streamlit Community Cloud

La versión que vamos a publicar es `07_rag_con_interfaz_streamlit.py`. No hace falta instalar Ollama en Community Cloud ni dejar tu computadora encendida: esta versión consulta la API de OpenAI.

### Preparar el repositorio

1. Creá un repositorio en GitHub y subí el código del proyecto.
2. Incluí `reglamento.pdf` junto al script y comprobá que sea un documento que puedas publicar.
3. Incluí `requirements.txt`, que ya contiene `streamlit`. Community Cloud instala las dependencias desde ese archivo; no usa las bibliotecas instaladas en tu computadora.
4. No subas `.env`, la carpeta `venv` ni ninguna clave.
5. Si creás un archivo local `.streamlit/secrets.toml`, no lo subas. El `.gitignore` del proyecto ya lo excluye.

`requirements.txt` fija las versiones de las dependencias directas del entorno revisado. Las dependencias transitivas no están fijadas y todavía hace falta verificar el despliegue real. Usá en el despliegue la misma versión de Python con la que verificaste la aplicación localmente.

### Crear la aplicación

1. Entrá a <https://share.streamlit.io/> y conectá tu cuenta de GitHub.
2. Elegí **Create app** y seleccioná el repositorio y la rama.
3. En el archivo principal o *Main file path*, ingresá:

   ```text
   07_rag_con_interfaz_streamlit.py
   ```

4. En **Advanced settings**, seleccioná una versión de Python compatible con tus dependencias.
5. En **Secrets**, agregá la configuración con formato TOML:

   ```toml
   OPENAI_API_KEY = "tu_clave_real"
   OPENAI_MODEL = "gpt-4.1-mini"
   OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
   ```

   No pegues el archivo `.env` sin adaptar su formato: en TOML los valores de texto van entre comillas. Dejá estas claves en la raíz, sin una sección entre corchetes; así Streamlit también las expone como variables de entorno y el `os.getenv()` del código puede leerlas.

6. Guardá y seleccioná **Deploy**.
7. Esperá la instalación y preparación inicial. Si falla, revisá los registros de la aplicación.

La aplicación publicada tendrá una URL para compartir. Podés actualizar los secretos desde su configuración sin guardarlos en GitHub.

Si ya tenías una versión publicada, subí los archivos actualizados a la rama del despliegue y agregá `OPENAI_EMBEDDING_MODEL` en **Settings → Secrets**. Si lo omitís se usará el valor predeterminado. Reiniciá la aplicación para reconstruir las colecciones con el modelo elegido.

### Qué tener en cuenta al publicarla

- No hace falta descargar modelos de Ollama ni de embeddings de Chroma. El servidor debe poder conectarse a la API de OpenAI tanto para embeddings como para respuestas.
- La base es temporal y se reconstruye al iniciar una sesión nueva o reiniciar la aplicación. No es almacenamiento persistente.
- Cada sesión prepara su propia colección y vuelve a pagar la generación de embeddings del documento. Para muchos usuarios o documentos grandes, convendría revisar memoria, caché, persistencia y procesamiento por lotes para respetar los límites de la API.
- Las consultas usan la clave del propietario de la aplicación y pueden generar cargos. Configurá controles de acceso y monitoreá el consumo antes de compartirla ampliamente; las alertas de presupuesto no deben asumirse como un bloqueo automático del gasto.
- Community Cloud tiene límites de recursos y no convierte esta demostración en un servicio de producción.
- El código, el PDF y las respuestas pueden exponer información. No publiques documentos privados en un repositorio público.

## 7. Problemas frecuentes

| Problema | Qué revisar |
| --- | --- |
| `ollama` no se reconoce | Completá la instalación y abrí otra terminal para actualizar el PATH. |
| No se puede conectar con Ollama | Abrí la aplicación Ollama. Si usás solamente la CLI, podés iniciar el servidor con `ollama serve`; no lo dupliques si ya está activo. |
| El modelo no existe | Revisá `ollama list` y descargá con `ollama pull` el nombre exacto usado en el script. |
| `ModuleNotFoundError` | Instalá `requirements.txt` usando el mismo Python con el que ejecutás el programa; incluye Streamlit. |
| Falta `OPENAI_API_KEY` | Verificá el nombre y ubicación de `.env`, o los Secrets del despliegue. |
| Error de autenticación o de cuota de OpenAI | Revisá la clave, los permisos del proyecto, el saldo y los límites de uso. |
| El PDF no devuelve texto | Confirmá que tenga texto seleccionable; si es escaneado, aplicá OCR. |
| La primera ejecución tarda | Puede estar procesándose el documento y generándose sus embeddings. En los pasos 06 y 07 se hacen llamadas a OpenAI; revisá también la conexión. |
| La respuesta es incorrecta | Mirá primero el contexto recuperado. Si falta el dato, puede fallar la recuperación; si el dato está, puede fallar la generación. |

## 8. Límites y próximos pasos

Esta propuesta prioriza que se vea el circuito completo sin sumar frameworks adicionales.

- La búsqueda por similitud siempre puede devolver un fragmento aunque no contenga la respuesta. En estos ejemplos no hay un umbral de relevancia.
- La instrucción de responder solo con el contexto no garantiza que el modelo nunca invente información.
- Los fragmentos se cortan por caracteres y pueden dividir una oración. Más adelante se puede mejorar la segmentación.
- Las bases se crean en memoria; no hay una etapa de indexación persistente separada de las consultas.
- Mostrar los fragmentos ayuda a inspeccionar las respuestas, pero todavía no hay citas con números de página.

Como ejercicio, probá una pregunta con respuesta explícita, otra sin respuesta y una que requiera combinar fragmentos. Compará siempre **documento → contexto recuperado → respuesta**.

## Documentación oficial

- Ollama en Windows: <https://docs.ollama.com/windows>
- API keys de OpenAI: <https://platform.openai.com/api-keys>
- Despliegue en Streamlit Community Cloud: <https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app>
- Secrets en Streamlit Community Cloud: <https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management>
