from pymongo import MongoClient
import os
from dotenv import load_dotenv
import json
import pytz
from datetime import datetime
import sqlite3
import matplotlib.pyplot as plt

CHARTS_FOLDER = os.path.join(os.getcwd(), 'Generador_plantillas', 'static', 'processed_charts')
if not os.path.exists(CHARTS_FOLDER):
    os.makedirs(CHARTS_FOLDER, exist_ok=True)


# Cargar variables de entorno
load_dotenv()

# Ruta de la base de datos
DB_PATH = './data/processed_datasets.db'

# Funciones de utilidad
def obtener_client():
    """Devuelve un cliente conectado a MongoDB."""
    uri = os.getenv("MONGODB_URI")  # Cadena de conexión
    return MongoClient(uri)

def obtener_client():
    uri = os.getenv("MONGODB_URI")  # Cadena de conexión
    return MongoClient(uri)

def obtener_db(nombre_db="call_center"):
    client = obtener_client()
    return client[nombre_db]

def registrar_mensaje(nombre, puesto, mensaje_usuario, respuesta_llm):
    historial_path = "historial_mensajes.json"
    # Cargar historial existente
    try:
        with open(historial_path, "r") as f:
            historial = json.load(f)
    except FileNotFoundError:
        historial = []
    # Agregar nuevo mensaje al historial
    historial.append({
        "nombre": nombre,
        "puesto": puesto,
        "mensaje_usuario": mensaje_usuario,
        "respuesta_llm": respuesta_llm,
        "fecha": datetime.now().isoformat()
    })
    # Guardar el historial actualizado
    with open(historial_path, "w") as f:
        json.dump(historial, f, indent=4)

def registrar_mensaje_mongo(nombre, puesto, mensaje_usuario, respuesta_llm):
    db = obtener_db()  # Conecta a la base de datos "call_center"
    coleccion = db["historial_mensajes"]  # Nombre de la colección
    # Crear documento
    documento = {
        "nombre": nombre,
        "puesto": puesto,
        "mensaje_usuario": mensaje_usuario,
        "respuesta_llm": respuesta_llm,
        "fecha": datetime.now().isoformat()
    }
    # Insertar en la colección
    coleccion.insert_one(documento)

def obtener_historial(nombre, puesto, max_mensajes=10):
    historial_path = "historial_mensajes.json"
    try:
        with open(historial_path, "r") as f:
            historial = json.load(f)
    except FileNotFoundError:
        return []
    # Filtrar mensajes del usuario actual y del puesto
    historial_filtrado = [
        h for h in historial if h["nombre"] == nombre and h["puesto"] == puesto
    ]
    # Limitar el número de mensajes a los más recientes
    return historial_filtrado[-max_mensajes:]

def obtener_historial_mongo(nombre, puesto, max_mensajes=10):
    db = obtener_db()  # Conecta a la base de datos "call_center"
    coleccion = db["historial_mensajes"]
    # Filtrar por usuario y puesto, ordenar por fecha descendente
    cursor = coleccion.find(
        {"nombre": nombre, "puesto": puesto},
        sort=[("fecha", -1)]  # Ordenar por fecha descendente
    ).limit(max_mensajes)
    return list(cursor)

def create_connection():
    """Crea y retorna una conexión a la base de datos."""
    if not os.path.exists('./data'):
        os.makedirs('./data')  # Crea el directorio si no existe
    conn = sqlite3.connect(DB_PATH)
    return conn

def initialize_db():
    """Inicializa la base de datos creando la tabla si no existe."""
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            category TEXT NOT NULL,
            processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            description TEXT,
            chart_paths TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_dataset(file_name, file_path, category, description, chart_paths, metricas=None):
    """Inserta un dataset procesado en la base de datos."""
    try:
        conn = sqlite3.connect('./data/processed_datasets.db')
        cursor = conn.cursor()

        # Ajustar rutas de los gráficos a formato relativo
        chart_paths = [
            path.replace('./static', '/static') if path.startswith('./static') else path
            for path in chart_paths
        ]
        chart_paths_str = ','.join(chart_paths)

        # Obtener hora actual en la zona horaria correcta
        timezone = pytz.timezone("Europe/Madrid")  # Cambia según tu ubicación
        current_time = datetime.now(timezone).strftime('%Y-%m-%d %H:%M:%S')

        query = """
        INSERT INTO datasets (file_name, file_path, category, description, created_at, chart_paths, metricas)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(query, (file_name, file_path, category, description, current_time, chart_paths_str, metricas))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"Error al insertar el dataset: {e}")

def get_all_datasets():
    """Recupera todos los datasets procesados desde la base de datos, incluidas las métricas."""
    try:
        conn = sqlite3.connect('./data/processed_datasets.db')
        cursor = conn.cursor()

        # Consulta para obtener todos los datasets
        query = """
        SELECT id, file_name, file_path, category, created_at, description, chart_paths, metricas
        FROM datasets
        ORDER BY created_at DESC
        """
        cursor.execute(query)
        datasets = cursor.fetchall()
        conn.close()
        # Convertir cada tupla en un diccionario
        return [
            {
                "id": row[0],
                "file_name": row[1],
                "file_path": row[2],
                "category": row[3],
                "created_at": row[4],
                "description": row[5],
                "chart_paths": row[6],
                "metricas": json.loads(row[7]) if row[7] else None,  # Deserializar métricas si existen
            }
            for row in datasets
        ]

    except sqlite3.Error as e:
        print(f"Error al obtener los datasets: {e}")
        return []


def get_dataset_by_id(dataset_id):
    """Recupera un dataset específico de la base de datos por su ID, incluidas las métricas."""
    try:
        conn = sqlite3.connect('./data/processed_datasets.db')
        cursor = conn.cursor()

        # Consulta para obtener el dataset por ID
        query = """
        SELECT id, file_name, file_path, category, created_at, description, chart_paths, metricas
        FROM datasets
        WHERE id = ?
        """
        cursor.execute(query, (dataset_id,))
        dataset = cursor.fetchone()

        conn.close()

        # Devuelve un diccionario con los datos si el dataset existe
        if dataset:
            return {
                "id": dataset[0],
                "file_name": dataset[1],
                "file_path": dataset[2],
                "category": dataset[3],
                "created_at": dataset[4],
                "description": dataset[5],
                "chart_paths": dataset[6],
                "metricas": json.loads(dataset[7]) if dataset[7] else None,  # Deserializar métricas si existen
            }
        return None
    except sqlite3.Error as e:
        print(f"Error al obtener el dataset por ID: {e}")
        return None

def save_chart(fig, filename):
    """
    Guarda un gráfico en el directorio especificado.
    :param fig: Figura de Matplotlib.
    :param filename: Nombre del archivo del gráfico.
    :return: Nombre del archivo guardado.
    """
    file_path = os.path.join(CHARTS_FOLDER, filename)
    fig.savefig(file_path, dpi=150)
    plt.close(fig)
    return filename  # Devolver solo el nombre del archivo

def update_analysis_in_db(dataset_id, analysis_text):
    """Actualiza el análisis generado en la base de datos para el dataset correspondiente."""
    try:
        conn = sqlite3.connect('./data/processed_datasets.db')
        cursor = conn.cursor()
        
        query = """
        UPDATE datasets
        SET analysis = ?
        WHERE id = ?
        """
        cursor.execute(query, (analysis_text, dataset_id))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"[ERROR] Error al actualizar el análisis en la base de datos: {e}")