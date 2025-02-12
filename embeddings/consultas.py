import os
from dotenv import load_dotenv
import pinecone
import openai
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

# Cargar variables de entorno
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))

# Realizar una búsqueda en el índice del rol
def buscar_documentos(query, rol, top_k=3):
    if rol not in pinecone.list_indexes():
        return f"No existe un índice para el rol {rol}."

    # Conectar al índice del rol
    embeddings = OpenAIEmbeddings()
    vectorstore = Pinecone(index_name=rol, embedding=embeddings)

    # Realizar la búsqueda
    embedding = openai.Embedding.create(
        input=query,
        model="text-embedding-ada-002"
    )["data"][0]["embedding"]

    resultados = vectorstore.similarity_search(embedding, k=top_k)
    return resultados
