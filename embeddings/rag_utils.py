import openai
import pinecone
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Cargar variables de entorno
load_dotenv()

# Crear cliente Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Función para conectar al índice
def conectar_indice(nombre_indice):
    if nombre_indice not in pc.list_indexes().names():
        pc.create_index(
            name=nombre_indice,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_ENVIRONMENT"))
        )
    return pc.Index(name=nombre_indice)

# Función para buscar documentos relevantes
def buscar_documentos(consulta, rol):
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    index = conectar_indice(rol.lower())
    
    resultados = index.query(
        vector=embeddings.embed_query(consulta),
        top_k=5,
        include_metadata=True
    )
    """
    # Depuración: Ver qué documentos se están recuperando
    print("Documentos recuperados:")
    for res in resultados["matches"]:
        print(f"- ID: {res['id']}, Título: {res['metadata'].get('titulo')}, Fragmento: {res['metadata'].get('fragmento', 'Sin contenido')}")
    """
    
    return resultados["matches"]  # Devuelve los fragmentos recuperados


# Crear un contexto con los documentos recuperados
def crear_contexto(resultados, query):
    documentos = "\n".join([f"Documento: {match['metadata'].get('titulo', 'Sin título')}.\n{match['metadata'].get('texto', 'Sin contenido.')}" for match in resultados["matches"]])
    contexto = f"""
    Información relevante recuperada:
    {documentos}

    Pregunta del usuario:
    {query}
    """
    return contexto

# Generar una respuesta con GPT-4
def generar_respuesta(contexto):
    prompt = f"""
    Usando la siguiente información, responde de manera clara y profesional:
    {contexto}
    """
    response = openai.ChatCompletion.create(
        model="chatgpt-4o-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"].strip()

# Flujo completo de RAG
def generar_respuesta_llm(consulta, fragmentos):
    # Construir el contexto con los fragmentos recuperados
    contexto_documentos = "\n".join(
        [f"- {frag['metadata'].get('fragmento', 'Sin contenido')}" for frag in fragmentos]
    )

    # Crear el prompt con los fragmentos relevantes
    prompt = f"""
    Basándote en la siguiente información relevante extraída de documentos:
    {contexto_documentos}

    Responde de manera clara y profesional a la siguiente consulta:
    {consulta}
    """
    # print(f"Prompt enviado al modelo:\n{prompt}")

    # Enviar el prompt al LLM
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error al generar la respuesta: {str(e)}"
    
# Flujo completo de RAG
def rag_respuesta(query, rol):
    resultados = buscar_documentos(query, rol)
    if isinstance(resultados, str):  # Si hay un error en la recuperación
        return resultados

    contexto = crear_contexto(resultados, query)
    respuesta = generar_respuesta(contexto)
    return respuesta