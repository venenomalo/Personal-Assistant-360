import os
from dotenv import load_dotenv
from openai import OpenAI

# Cargar variables de entorno
load_dotenv()

# Inicializar el cliente OpenAI
client = OpenAI()

def plantillas_csr_escalacion_bo(tipologia, motivo, cliente, detalles):
    """
    Genera una plantilla para escalar un caso a Back Office utilizando un LLM (GPT-4).
    """
    prompt = f"""
    Actúa como un asistente para un teleoperador de atención al cliente de una compañía telefónica. Genera una plantilla detallada para escalar un caso a otro CSR en el equipo de Back Office. La plantilla debe incluir la información necesaria para que el CSR comprenda el caso de manera clara y pueda resolverlo eficientemente. Basado en la siguiente información:
    - Tipología: {tipologia}.
    - Motivo: {motivo}.
    - Nombre del cliente: {cliente}.
    - Detalles: {detalles}.

    La plantilla debe ser clara, organizada y sencilla. Es una plantilla para copiarse posteriormente en una herramienta de ticketing.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Eres un asistente experto en atención al cliente."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error al generar la plantilla: {str(e)}"

def plantillas_csr_respuesta_cliente(tipologia, motivo, cliente, detalles):
    """
    Genera una plantilla para responder a un cliente sobre su consulta utilizando un LLM (GPT-4).
    """
    prompt = f"""
    Actúa como un asistente para un teleoperador de atención al cliente de una compañía telefónica. Genera una plantilla profesional para responder a un cliente sobre su consulta. La respuesta debe ser clara, organizada y contener toda la información relevante. Basado en la siguiente información:
    - Tipología: {tipologia}.
    - Motivo: {motivo}.
    - Nombre del cliente: {cliente}.
    - Detalles: {detalles}.

    La respuesta debe reflejar empatía y profesionalismo. La plantilla es para enviarse por mail.
    """

    try:
        response = client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=[
                {"role": "system", "content": "Eres un asistente experto en atención al cliente."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error al generar la plantilla: {str(e)}"