import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
import PyPDF2
import tiktoken
from rag_utils import conectar_indice
import fitz

# Cargar variables de entorno
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

# Inicializar Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Función para extraer texto de PDFs
def extraer_texto(pdf_path):
    texto = ""
    try:
        with fitz.open(pdf_path) as doc:
            for pagina in doc:
                texto += pagina.get_text()
    except Exception as e:
        print(f"Error al procesar {pdf_path} con PyMuPDF: {e}")
    return texto

# Dividir texto en fragmentos manejables
def dividir_texto(texto, max_tokens=1000):
    tokenizer = tiktoken.get_encoding("cl100k_base")  # Modelo compatible con text-embedding-ada-002
    tokens = tokenizer.encode(texto)
    fragmentos = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return ["".join(tokenizer.decode(fragmento)) for fragmento in fragmentos]

# Procesar un PDF y subir los embeddings al índice correspondiente
def procesar_y_guardar_embeds(pdf_path, metadata, rol):
    texto = extraer_texto(pdf_path)
    fragmentos = dividir_texto(texto)
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    # Crear o conectar al índice
    if rol not in pc.list_indexes().names():
        pc.create_index(
            name=rol,
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region=pinecone_environment)
        )
    index = pc.Index(name=rol)  # Conectar al índice

    # Subir cada fragmento al índice
    for i, fragmento in enumerate(fragmentos):
        embedding = embeddings.embed_query(fragmento)
        cleaned_metadata = {
            "id": metadata["id"],
            "roles": metadata.get("roles", []),
            "titulo": metadata.get("titulo", "Sin título"),
            "fragmento": fragmento  # Agregar el contenido del fragmento
        }
        print(f"Subiendo fragmento {i + 1}/{len(fragmentos)} del documento {metadata['id']} al índice {rol}")
        try:
            index.upsert(
                vectors=[
                    {
                        "id": f"{metadata['id']}-{i}",
                        "values": embedding,
                        "metadata": cleaned_metadata,
                    }
                ]
            )
        except Exception as e:
            print(f"Error al subir el vector con ID {metadata['id']}-{i}: {e}")

# Procesar todos los PDFs en una carpeta específica para cada rol
def procesar_carpeta(carpeta, rol):
    carpeta = os.path.join("embeddings", "data", rol)  # Ajustar ruta
    if not os.path.exists(carpeta):
        print(f"La carpeta {carpeta} no existe. Omitiendo...")
        return
    for archivo in os.listdir(carpeta):
        if archivo.endswith(".pdf"):
            pdf_path = os.path.join(carpeta, archivo)
            metadata = {"id": archivo.split(".")[0], "rol": rol}
            procesar_y_guardar_embeds(pdf_path, metadata, rol)
            print(f"Procesado: {archivo}")

# Procesar documentos en la carpeta "general" y subirlos al índice de RRHH
def procesar_carpeta_general():
    carpeta = os.path.join("embeddings", "data", "general")
    if not os.path.exists(carpeta):
        print(f"La carpeta {carpeta} no existe. Omitiendo...")
        return
    for archivo in os.listdir(carpeta):
        if archivo.endswith(".pdf"):
            pdf_path = os.path.join(carpeta, archivo)
            metadata = {
                "id": archivo.split(".")[0],
                "roles": ["rrhh", "supervisor", "planificacion"],  # Roles que tendrán acceso
                "titulo": archivo.split(".")[0].replace("_", " ")  # Título basado en el nombre del archivo
            }
            for rol in ["rrhh", "supervisor", "planificacion"]:
                procesar_y_guardar_embeds(pdf_path, metadata, rol)
            print(f"Procesado documento compartido: {archivo}")
            
# Ejecutar el procesamiento
if __name__ == "__main__":
    # Procesar carpetas específicas para cada rol
    roles = ["csr", "supervisor", "rrhh", "planificacion", "reporting"]
    for rol in roles:
        carpeta = f"data/{rol}"  # Ajusta según tu estructura
        procesar_carpeta(carpeta, rol)

    # Procesar carpeta general y subir al índice RRHH
    procesar_carpeta_general()
