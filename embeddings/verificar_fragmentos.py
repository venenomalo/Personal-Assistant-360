import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Cargar variables de entorno
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

# Inicializar Pinecone
pc = Pinecone(api_key=pinecone_api_key)

index = pc.Index("rrhh")
resultados = index.query(
    vector=[0]*1536,  # Consulta vacía para devolver fragmentos
    top_k=10,
    include_metadata=True
)

print("Fragmentos disponibles:")
for res in resultados["matches"]:
    print(f"ID: {res['id']}, Roles: {res['metadata'].get('roles')}, Título: {res['metadata'].get('titulo')}, Fragmento: {res['metadata'].get('fragmento', 'No disponible')}")