from flask import Flask, render_template, request, session, redirect, url_for, flash, send_file, make_response, send_file
from werkzeug.utils import secure_filename
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import sys
import os
import shutil
import time
import json
import sqlite3
# Obtiene el directorio actual donde está app.py
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# Asegurar que el directorio raíz del proyecto está en sys.path
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))
# Asegurar que el directorio de Generador_plantillas está en sys.path
GENERATOR_DIR = os.path.join(BASE_DIR, "Generador_plantillas")
if os.path.exists(GENERATOR_DIR):
    sys.path.append(GENERATOR_DIR)
from Generador_plantillas.plantilla_llm import plantillas_csr_escalacion_bo, plantillas_csr_respuesta_cliente
from embeddings.rag_utils import rag_respuesta
from embeddings.rag_utils import buscar_documentos
from Generador_plantillas.data_processing import process_nds_servicio, process_nps_agente
from Generador_plantillas.db_utils import obtener_client, obtener_db, registrar_mensaje_mongo, obtener_historial_mongo, insert_dataset, get_all_datasets, get_dataset_by_id, update_analysis_in_db
from Generador_plantillas.report_generator import export_to_word_simple
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


# Cargar variables de entorno
load_dotenv()
# Configuración básica
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuración del directorio de carga de archivos
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Crear carpeta si no existe
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


"""
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
    print(f"Mensaje registrado para {nombre} ({puesto})")

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

def obtener_contexto_por_puesto(puesto):
    # Devuelve el contexto basado en el puesto del usuario.
    contextos = {
        "CSR": "Eres un asistente experto en atención al cliente en un call center...",
        "Planificacion": "Eres un asistente especializado en planificación y administración...",
        "Supervisor": "Eres un asistente para supervisores y coordinadores de equipos...",
        "RRHH": "Eres un asistente para el departamento de Recursos Humanos...",
        "Reporting": "Eres un asistente experto en reporting y análisis de datos...",
    }
    return contextos.get(puesto, "Eres un asistente genérico.")
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        nombre = request.form["nombre"]
        puesto = request.form["puesto"]

        # Guardar en la sesión
        session["nombre"] = nombre
        session["puesto"] = puesto

        return redirect(url_for("interaccion_llm"))
    return render_template("index.html")

@app.route("/historial", methods=["GET"])
def historial():
    if "nombre" not in session or "puesto" not in session:
        return redirect(url_for("index"))

    nombre = session["nombre"]
    puesto = session["puesto"]

    # Obtener historial desde MongoDB
    historial = obtener_historial_mongo(nombre, puesto)

    return render_template("historial.html", historial=historial)

"""
def historial():
    if "nombre" not in session or "puesto" not in session:
        return redirect(url_for("index"))

    nombre = session["nombre"]
    puesto = session["puesto"]

    # Filtrar historial por usuario y rol
    try:
        with open("historial_mensajes.json", "r") as f:
            historial = json.load(f)
    except FileNotFoundError:
        historial = []

    historial_filtrado = [
        h for h in historial if h["nombre"] == nombre and h["puesto"] == puesto
    ]

    return render_template("historial.html", historial=historial_filtrado)
"""


@app.route("/interaccion_llm", methods=["GET", "POST"])
def interaccion_llm():
    if "nombre" not in session or "puesto" not in session:
        return redirect(url_for("index"))

    nombre = session["nombre"]
    puesto = session["puesto"]
    respuesta = None

    if request.method == "POST":
        pregunta = request.form["pregunta"]
        
        # Recuperar historial del usuario actual
        historial = obtener_historial_mongo(nombre, puesto)
        # historial = obtener_historial(nombre, puesto)

        # Formatear historial para el prompt
        contexto_historial = "\n".join(
            [f"- Usuario: {h['mensaje_usuario']} | Respuesta: {h['respuesta_llm']}" for h in historial]
        )

        # Personalizar el contexto según el puesto
        if puesto == "CSR":
            contexto_puesto = "Eres un asistente experto en atención al cliente en un call center. Ayudas a los agentes (CSR) a manejar procesos de facturación, resolver disputas de clientes, gestionar casos de activación o desactivación de servicios, y proporcionar orientación en la resolución de consultas de clientes. Además, ayudas en la redacción de mensajes claros y profesionales para los clientes y en la creación de plantillas para escalación de casos al Back Office."
        elif puesto == "Planificacion":
            contexto_puesto = "Eres un asistente especializado en planificación y administración en un call center. Ayudas con la organización y planificación de horarios de turnos, asignación de descansos, y la gestión de solicitudes de vacaciones. También puedes generar reportes sobre la disponibilidad del personal, planificar coberturas en horarios pico, y asegurarte de que los recursos estén optimizados para cumplir con los objetivos operativos."
        elif puesto == "Supervisor":
            contexto_puesto = "Eres un asistente para supervisores y coordinadores de equipos en un call center. Ayudas en la optimización de equipos mediante estrategias basadas en datos, análisis de KPIs como NPS, AHT, FCR y rellamadas. Proporcionas recomendaciones para mejorar el desempeño del equipo y herramientas para coaching individual. También ayudas en la generación de reportes para justificar decisiones operativas y apoyar a los CSR en sus dudas más complejas."
        elif puesto == "RRHH":
            contexto_puesto = "Eres un asistente para el departamento de Recursos Humanos en un call center. Ayudas con la gestión de permisos, solicitudes de vacaciones, bajas laborales y contrataciones. Proporcionas información sobre políticas laborales, normativas vigentes, y guías sobre procesos internos. También puedes generar cartas y plantillas formales, como cartas de aviso, acuerdos de horario, y documentos para procesos disciplinarios."
        elif puesto == "Reporting":
            contexto_puesto = "Eres un asistente experto en reporting y análisis de datos en un call center. Ayudas en la generación de dashboards claros y efectivos utilizando datos operativos. Puedes procesar grandes volúmenes de datos, identificar tendencias clave, y generar reportes visuales que resalten el rendimiento de KPIs. También ayudas a automatizar reportes repetitivos y ofrecer insights que apoyen la toma de decisiones estratégicas."
        else:
            contexto_puesto = "Eres un asistente genérico."

        # Recuperar documentos relevantes desde Pinecone
        try:
            resultados = buscar_documentos(pregunta, puesto)  # Usar función de recuperación de Pinecone
            if resultados:
                contexto_documentos = "\n".join(
                    [f"- {match['metadata'].get('fragmento', 'Sin contenido')}" for match in resultados]
                )
            else:
                contexto_documentos = "No se encontraron documentos relevantes para tu consulta."
        except Exception as e:
            contexto_documentos = f"Error al recuperar información: {str(e)}"

        # Crear el prompt final
        prompt = f"""

        Basándote en la siguiente información relevante para el rol de {puesto}:
        {contexto_puesto}

        Historial de mensajes recientes:
        {contexto_historial}

        Información adicional recuperada:
        {contexto_documentos}

        Responde de manera clara y profesional a la consulta:
        {pregunta}
        """
        
        # Generar respuesta desde el modelo
        try:
            response = client.chat.completions.create(
                model="chatgpt-4o-latest",
                messages=[{"role": "user", "content": prompt}]
            )
            respuesta = response.choices[0].message.content.strip()
        except Exception as e:
            respuesta = f"Error al generar la respuesta: {str(e)}"
        
        # Registrar el mensaje en el historial
        registrar_mensaje_mongo(nombre, puesto, pregunta, respuesta)
        # registrar_mensaje(nombre, puesto, pregunta, respuesta)

    return render_template("interaccion_llm.html", nombre=nombre, puesto=puesto, respuesta=respuesta)


@app.route("/plantillas_csr", methods=["GET", "POST"])
def plantillas_csr():
    if "nombre" not in session or "puesto" not in session:
        return redirect(url_for("index"))

    nombre = session["nombre"]
    puesto = session["puesto"]

    if puesto != "CSR":
        return "No tienes acceso a esta sección.", 403

    plantilla = None

    # Mapeo de tipos de plantillas a funciones específicas
    plantilla_funciones = {
        "escalar_bo": plantillas_csr_escalacion_bo,
        "respuesta_cliente": plantillas_csr_respuesta_cliente,
    }

    if request.method == "POST":
        tipo_plantilla = request.form["tipo_plantilla"]
        tipologia = request.form["tipologia"]
        motivo = request.form["motivo"]
        cliente = request.form["cliente"]
        detalles = request.form["detalles"]

        # Validar y ejecutar la función correspondiente
        if tipo_plantilla in plantilla_funciones:
            plantilla = plantilla_funciones[tipo_plantilla](tipologia, motivo, cliente, detalles)
        else:
            plantilla = "Tipo de plantilla no válido."

    return render_template("plantillas_csr.html", plantilla=plantilla, nombre=nombre)
"""
@app.route('/upload', methods=['POST'])
def upload_file():
    """"""Endpoint para manejar la carga de archivos.""""""
    if 'file' not in request.files:
        flash('No se seleccionó ningún archivo.', 'error')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No se seleccionó ningún archivo válido.', 'error')
        return redirect(url_for('index'))

    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], "last_uploaded_file.csv")
        if file.filename.endswith('.csv'):
            file.save(file_path)
        elif file.filename.endswith('.xlsx'):
            data = pd.read_excel(file)
            data.to_csv(file_path, index=False)
        else:
            flash('Formato no soportado. Solo se aceptan archivos CSV o Excel.', 'error')
            return redirect(url_for('index'))

        flash('Archivo cargado correctamente.', 'success')
        print("Redirigiendo a /view_data")
        return redirect(url_for('view_data'))

    except Exception as e:
        flash(f'Error al procesar el archivo: {e}', 'error')
        print("Error en /upload:", e)
        return redirect(url_for('index'))

@app.route('/view_data', methods=['GET'])
def view_data():
    """"""Mostrar datos del archivo cargado.""""""
    try:
        # Ruta del archivo cargado
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], "last_uploaded_file.csv")
        if not os.path.exists(file_path):
            flash("No se encontró el archivo cargado. Por favor, sube un archivo.", "error")
            return redirect(url_for('index'))

        # Leer el archivo
        data = pd.read_csv(file_path)

        # Identificar tipo de archivo según las columnas
        columnas = list(data.columns)
        if 'Agent' in columnas and 'Satisfaction rating' in columnas:
            tipo_archivo = "NPS_Agentes"
        elif 'Index' in columnas and 'Service Level (20 Seconds)' in columnas:
            tipo_archivo = "NDS_Servicio"
        else:
            flash("El archivo cargado no coincide con ningún formato esperado.", "error")
            return redirect(url_for('index'))

        # Convertir vista previa a HTML
        preview = data.head(10).to_html(classes='table table-striped', index=False)

        # Información básica
        info = {
            "tipo_archivo": tipo_archivo,
            "num_filas": len(data),
            "num_columnas": len(data.columns),
            "columnas": columnas,
        }

        print(f"Archivo identificado como: {tipo_archivo}")
        return render_template('view_data.html', preview=preview, info=info, tipo_archivo=tipo_archivo)

    except Exception as e:
        flash(f"Error al mostrar los datos: {e}", "error")
        return redirect(url_for('index'))
    
@app.route('/process_data', methods=['GET'])
def process_data():
    """"""Procesar datos cargados y generar gráficos específicos.""""""
    try:
        # Ruta del archivo cargado
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], "last_uploaded_file.csv")
        if not os.path.exists(file_path):
            flash("No se encontró el archivo cargado. Por favor, sube un archivo.", "error")
            return redirect(url_for('index'))

        # Leer el archivo
        data = pd.read_csv(file_path)

        # Identificar tipo de archivo según las columnas
        columnas = list(data.columns)
        if 'Service Level (20 Seconds)' in columnas and 'Incoming Calls' in columnas:
            tipo_archivo = "NDS_Servicio"

            if 'Service Level (20 Seconds)' in data.columns and 'Incoming Calls' in data.columns:
                # Convertir columnas con formato HH:MM:SS a segundos
                for column in ['Waiting Time (AVG)', 'Talk Duration (AVG)']:
                    if column in data.columns:
                        data[column] = pd.to_timedelta(data[column]).dt.total_seconds()

                # Limpiar y convertir 'Service Level (20 Seconds)' a número
                data['Service Level (20 Seconds)'] = (
                    data['Service Level (20 Seconds)']
                    .str.replace('%', '', regex=False)  # Eliminar el símbolo '%'
                    .astype(float) / 100  # Convertir a porcentaje en formato decimal
                )
                data['Incoming Calls'] = pd.to_numeric(data['Incoming Calls'], errors='coerce')

                # Manejar valores nulos
                data.dropna()

                # Cálculo del NDS del servicio
                data['Weighted_Service_Level'] = data['Service Level (20 Seconds)'] * data['Incoming Calls']
                total_incoming_calls = data['Incoming Calls'].sum()

                if total_incoming_calls > 0:
                    nds_servicio = data['Weighted_Service_Level'].sum() / total_incoming_calls
                else:
                    nds_servicio = 0

                print(f"NDS del Servicio calculado: {nds_servicio}")

                # 1. Gráfico de Barras: Contestadas vs Abandonadas
                fig1, ax1 = plt.subplots()
                data[['Answered Calls', 'Abandoned Calls']].sum().plot(kind='bar', ax=ax1, color=['green', 'red'])
                ax1.set_title('Llamadas Contestadas vs Abandonadas')
                ax1.set_ylabel('Número de Llamadas')
                buf1 = io.BytesIO()
                plt.savefig(buf1, format="png")
                plt.close(fig1)
                buf1.seek(0)
                calls_chart_url = base64.b64encode(buf1.getvalue()).decode()
                buf1.close()

                # 2. Gráfico de Línea: Evolución del Nivel de Servicio
                fig2, ax2 = plt.subplots()
                data.plot(x='Index', y='Service Level (20 Seconds)', kind='line', ax=ax2, color='blue', linewidth=2)
                ax2.set_title('Evolución del Nivel de Servicio (20s)')
                ax2.set_xlabel('Índice')
                ax2.set_ylabel('Nivel de Servicio (%)')
                ax2.grid(True, linestyle='--', alpha=0.7)
                buf2 = io.BytesIO()
                plt.savefig(buf2, format="png")
                plt.close(fig2)
                buf2.seek(0)
                service_level_chart_url = base64.b64encode(buf2.getvalue()).decode()
                buf2.close()

                # 3. Gráfico de Dispersión: Nivel de Servicio vs Tiempo de Espera
                fig3, ax3 = plt.subplots()
                data.plot.scatter(x='Service Level (20 Seconds)', y='Waiting Time (AVG)', ax=ax3, color='purple')
                ax3.set_title('Nivel de Servicio vs Tiempo de Espera')
                ax3.set_xlabel('Nivel de Servicio (%)')
                ax3.set_ylabel('Tiempo de Espera Promedio (segundos)')
                buf3 = io.BytesIO()
                plt.savefig(buf3, format="png")
                plt.close(fig3)
                buf3.seek(0)
                service_vs_waiting_chart_url = base64.b64encode(buf3.getvalue()).decode()
                buf3.close()

                # 4. Gráfico de Barras: Talk Duration (AVG)
                fig4, ax4 = plt.subplots()
                data['Talk Duration (AVG)'].plot(kind='hist', bins=20, ax=ax4, color='orange', alpha=0.7)
                ax4.set_title('Distribución de la Duración Promedio de Conversación')
                ax4.set_xlabel('Duración (segundos)')
                ax4.set_ylabel('Frecuencia')
                buf4 = io.BytesIO()
                plt.savefig(buf4, format="png")
                plt.close(fig4)
                buf4.seek(0)
                talk_duration_chart_url = base64.b64encode(buf4.getvalue()).decode()
                buf4.close()

                metricas = {
                    "NDS del Servicio": round(nds_servicio, 2),
                    "Llamadas Entrantes": total_incoming_calls,
                    "Llamadas Contestadas": data['Answered Calls'].sum(),
                    "Llamadas Abandonadas": data['Abandoned Calls'].sum(),
                }

                # Renderizar plantilla
                return render_template(
                    'process_data.html',
                    tipo_archivo="NDS_Servicio",
                    calls_chart_url=calls_chart_url,
                    service_level_chart_url=service_level_chart_url,
                    service_vs_waiting_chart_url=service_vs_waiting_chart_url,
                    talk_duration_chart_url=talk_duration_chart_url,
                    metricas=metricas
                )


        elif 'Agent' in columnas and 'Satisfaction rating' in columnas:
            tipo_archivo = "NPS_Agente"

            # Convertir Satisfaction Rating a NPS
            if 'Satisfaction rating' in data.columns and 'Answered (Y/N)' in data.columns:
                # Filtrar filas con respuestas válidas
                data_valid = data[data['Answered (Y/N)'] == 'Y'].copy()  # Asegura que sea una copia

                # Convertir Satisfaction rating a escala 0-10
                data_valid.loc[:, 'Satisfaction rating'] = data_valid['Satisfaction rating'] * 2

                # Clasificar como Promotores, Detractores y Pasivos
                data_valid.loc[:, 'NPS'] = data_valid['Satisfaction rating'].apply(
                    lambda x: 100 if x >= 9 else (-100 if x <= 6 else 0)
                )

                # Convertir AvgTalkDuration a segundos
                if 'AvgTalkDuration' in data_valid.columns:
                    data_valid['AvgTalkDuration'] = pd.to_timedelta(data_valid['AvgTalkDuration']).dt.total_seconds()

                # Calcular métricas
                nps_promoters = len(data_valid[data_valid['NPS'] == 100])
                nps_detractors = len(data_valid[data_valid['NPS'] == -100])
                nps_passives = len(data_valid[(data_valid['NPS'] != 100) & (data_valid['NPS'] != -100)])
                total_responses = len(data_valid)

                if total_responses > 0:
                    nps_final = ((nps_promoters - nps_detractors) / total_responses) * 100
                else:
                    nps_final = 0

                print(f"NPS calculado: {nps_final:.2f}")

                # 1. Gráfico de Pastel: Distribución de Promotores, Pasivos y Detractores
                fig1, ax1 = plt.subplots()
                labels = ['Promotores', 'Pasivos', 'Detractores']
                sizes = [nps_promoters, nps_passives, nps_detractors]
                colors = ['green', 'yellow', 'red']
                ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
                ax1.axis('equal')  # Asegura que el gráfico sea circular
                buf1 = io.BytesIO()
                plt.savefig(buf1, format="png")
                plt.close(fig1)
                buf1.seek(0)
                pie_chart_url = base64.b64encode(buf1.getvalue()).decode()
                buf1.close()

                # 2. Gráfico de Barras: NPS por Tema (Topic)
                if 'Topic' in data_valid.columns:
                    fig2, ax2 = plt.subplots()
                    data_valid.groupby('Topic')['NPS'].mean().plot(kind='bar', ax=ax2, color='skyblue')
                    ax2.set_title('NPS Promedio por Tema')
                    ax2.set_ylabel('NPS')
                    ax2.set_xlabel('Tema')
                    buf2 = io.BytesIO()
                    plt.savefig(buf2, format="png")
                    plt.close(fig2)
                    buf2.seek(0)
                    topic_chart_url = base64.b64encode(buf2.getvalue()).decode()
                    buf2.close()
                else:
                    topic_chart_url = None

                # 3. Gráfico de Barras: Promedio de Velocidad de Respuesta por Categoría
                if 'Speed of answer in seconds' in data_valid.columns:
                    fig3, ax3 = plt.subplots()
                    data_valid.groupby('NPS')['Speed of answer in seconds'].mean().plot(kind='bar', ax=ax3, color='orange')
                    ax3.set_title('Velocidad de Respuesta por Categoría de NPS')
                    ax3.set_xlabel('Categoría de NPS')
                    ax3.set_ylabel('Velocidad de Respuesta (segundos)')
                    buf3 = io.BytesIO()
                    plt.savefig(buf3, format="png")
                    plt.close(fig3)
                    buf3.seek(0)
                    speed_chart_url = base64.b64encode(buf3.getvalue()).decode()
                    buf3.close()
                else:
                    speed_chart_url = None

                # 4. Gráfico de Barras: Duración Promedio de Conversación por Categoría
                if 'AvgTalkDuration' in data_valid.columns:
                    fig4, ax4 = plt.subplots()
                    data_valid.groupby('NPS')['AvgTalkDuration'].mean().plot(kind='bar', ax=ax4, color='purple')
                    ax4.set_title('Duración Promedio de Conversación por Categoría de NPS')
                    ax4.set_xlabel('Categoría de NPS')
                    ax4.set_ylabel('Duración Promedio (segundos)')
                    buf4 = io.BytesIO()
                    plt.savefig(buf4, format="png")
                    plt.close(fig4)
                    buf4.seek(0)
                    duration_chart_url = base64.b64encode(buf4.getvalue()).decode()
                    buf4.close()
                else:
                    duration_chart_url = None

                # 5. Gráfico de Barras: NPS por Agente
                if 'Agent' in data_valid.columns:
                    fig5, ax5 = plt.subplots()
                    data_valid.groupby('Agent')['NPS'].mean().plot(kind='bar', ax=ax5, color='blue')
                    ax5.set_title('NPS Promedio por Agente')
                    ax5.set_ylabel('NPS')
                    ax5.set_xlabel('Agente')
                    buf5 = io.BytesIO()
                    plt.savefig(buf5, format="png")
                    plt.close(fig5)
                    buf5.seek(0)
                    agent_chart_url = base64.b64encode(buf5.getvalue()).decode()
                    buf5.close()
                else:
                    agent_chart_url = None

                # Métricas para mostrar
                metricas = {
                    "NPS Global": round(nps_final, 2),
                    "Promotores": nps_promoters,
                    "Pasivos": nps_passives,
                    "Detractores": nps_detractors,
                    "Total de Respuestas": total_responses,
                }

                # Renderizar plantilla con gráficos
                return render_template(
                    'process_data.html',
                    tipo_archivo="NPS_Agente",
                    pie_chart_url=pie_chart_url,
                    topic_chart_url=topic_chart_url,
                    speed_chart_url=speed_chart_url,
                    duration_chart_url=duration_chart_url,
                    agent_chart_url=agent_chart_url,
                    metricas=metricas
                )

        else:
            flash("El archivo cargado no coincide con ningún formato esperado.", "error")
            return redirect(url_for('index'))

        return render_template('process_data.html', tipo_archivo=tipo_archivo, plot_url=plot_url, metricas=metricas)

    except Exception as e:
        flash(f"Error al procesar los datos: {e}", "error")
        print(f"Error en /process_data: {e}")
        return redirect(url_for('index'))
"""
@app.route('/upload', methods=['POST'])
def upload_file():
    """Endpoint para manejar la carga de archivos."""
    if 'file' not in request.files:
        flash('No se seleccionó ningún archivo.', 'error')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No se seleccionó ningún archivo válido.', 'error')
        return redirect(url_for('index'))

    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], "last_uploaded_file.csv")
        if file.filename.endswith('.csv'):
            file.save(file_path)
        elif file.filename.endswith('.xlsx'):
            data = pd.read_excel(file)
            data.to_csv(file_path, index=False)
        else:
            flash('Formato no soportado. Solo se aceptan archivos CSV o Excel.', 'error')
            return redirect(url_for('index'))

        flash('Archivo cargado correctamente.', 'success')
        print("Redirigiendo a /view_data")
        return redirect(url_for('view_data'))

    except Exception as e:
        flash(f'Error al procesar el archivo: {e}', 'error')
        print("Error en /upload:", e)
        return redirect(url_for('index'))


@app.route('/view_data', methods=['GET'])
def view_data():
    """Mostrar datos del archivo cargado."""
    try:
        # Ruta del archivo cargado
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], "last_uploaded_file.csv")
        if not os.path.exists(file_path):
            flash("No se encontró el archivo cargado. Por favor, sube un archivo.", "error")
            return redirect(url_for('index'))

        # Leer el archivo
        data = pd.read_csv(file_path)

        # Identificar tipo de archivo según las columnas
        columnas = list(data.columns)
        if 'Agent' in columnas and 'Satisfaction rating' in columnas:
            tipo_archivo = "NPS_Agentes"
        elif 'Index' in columnas and 'Service Level (20 Seconds)' in columnas:
            tipo_archivo = "NDS_Servicio"
        else:
            flash("El archivo cargado no coincide con ningún formato esperado.", "error")
            return redirect(url_for('index'))

        # Convertir vista previa a HTML
        preview = data.head(5).to_html(classes='table table-striped', index=False)

        # Información básica
        info = {
            "tipo_archivo": tipo_archivo,
            "num_filas": len(data),
            "num_columnas": len(data.columns),
            "columnas": columnas,
        }

        print(f"Archivo identificado como: {tipo_archivo}")
        return render_template('view_data.html', preview=preview, info=info, tipo_archivo=tipo_archivo)

    except Exception as e:
        flash(f"Error al mostrar los datos: {e}", "error")
        return redirect(url_for('index'))

"""
@app.route('/process_data', methods=['GET'])
def process_data():
    """"""Procesar datos cargados y generar gráficos específicos.""""""
    try:
        # Ruta del archivo cargado
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], "last_uploaded_file.csv")
        if not os.path.exists(file_path):
            flash("No se encontró el archivo cargado. Por favor, sube un archivo.", "error")
            return redirect(url_for('index'))

        # Leer el archivo
        data = pd.read_csv(file_path)

        # Identificar tipo de archivo según las columnas
        columnas = list(data.columns)
        if 'Service Level (20 Seconds)' in columnas and 'Incoming Calls' in columnas:
            from data_processing import process_nds_servicio
            nds_servicio, charts = process_nds_servicio(data)
            return render_template(
                'process_data.html',
                tipo_archivo="NDS_Servicio",
                metricas={
                    "NDS del Servicio": round(nds_servicio, 2),
                    "Llamadas Entrantes": data['Incoming Calls'].sum(),
                    "Llamadas Contestadas": data['Answered Calls'].sum(),
                    "Llamadas Abandonadas": data['Abandoned Calls'].sum(),
                },
                calls_chart=charts.get('calls_chart'),
                service_level_chart=charts.get('service_level_chart'),
                service_vs_waiting_chart=charts.get('service_vs_waiting_chart'),
                talk_duration_chart=charts.get('talk_duration_chart')
            )

        elif 'Agent' in columnas and 'Satisfaction rating' in columnas:
            from data_processing import process_nps_agente
            nps_final, charts = process_nps_agente(data)
            metricas = {
                "NPS Global": round(nps_final, 2),
            }
            print("Gráficos enviados al template:", charts.keys()) # Debug
            return render_template(
                'process_data.html',
                tipo_archivo="NPS_Agente",
                metricas=metricas,
                **charts
            )

        else:
            flash("El archivo cargado no coincide con ningún formato esperado.", "error")
            return redirect(url_for('index'))

    except Exception as e:
        flash(f"Error al procesar los datos: {e}", "error")
        print(f"Error en /process_data: {e}")
        return redirect(url_for('index'))
"""

@app.route('/process_data', methods=['GET'])
def process_data():
    """Procesar datos cargados, generar gráficos y guardar en la base de datos."""
    try:
        # Ruta del archivo cargado
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], "last_uploaded_file.csv")
        if not os.path.exists(file_path):
            flash("No se encontró el archivo cargado. Por favor, sube un archivo.", "error")
            return redirect(url_for('index'))

        # Leer el archivo
        data = pd.read_csv(file_path)

        # Identificar tipo de archivo según las columnas
        columnas = list(data.columns)
        if 'Service Level (20 Seconds)' in columnas and 'Incoming Calls' in columnas:
            nds_servicio, charts, processed_data, metricas = process_nds_servicio(data)

            # Guardar los datos procesados en lugar del archivo original
            file_name = "NDS_Servicio_Procesado.csv"
            processed_file_path = f"./data/processed_files/{file_name}"
            os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
            processed_data.to_csv(processed_file_path, index=False)

            description = "Reporte procesado de nivel de servicio."
            # Convertir las métricas a tipos estándar de Python
            processed_metricas = {key: (value.item() if hasattr(value, "item") else value) for key, value in metricas.items()}

            insert_dataset(file_name, processed_file_path, "NDS_Servicio", description, list(charts.values()), metricas=json.dumps(processed_metricas))


            return render_template(
                'process_data.html',
                tipo_archivo="NDS_Servicio",
                metricas={
                    "NDS del Servicio": round(nds_servicio, 2),
                    "Llamadas Entrantes": metricas.get("Llamadas Entrantes", 0),
                    "Llamadas Contestadas": metricas.get("Llamadas Contestadas", 0),
                    "Llamadas Abandonadas": metricas.get("Llamadas Abandonadas", 0),
                },
                calls_chart=charts.get('calls_chart'),
                service_level_chart=charts.get('service_level_chart'),
                service_vs_waiting_chart=charts.get('service_vs_waiting_chart'),
                talk_duration_chart=charts.get('talk_duration_chart')
            )

        elif 'Agent' in columnas and 'Satisfaction rating' in columnas:
            nps_final, charts, processed_data, metricas = process_nps_agente(data)

            # Guardar los datos procesados en lugar del archivo original
            file_name = "NPS_Agente_Procesado.csv"
            processed_file_path = f"./data/processed_files/{file_name}"
            os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
            processed_data.to_csv(processed_file_path, index=False)

            description = "Reporte procesado de análisis de satisfacción."
            # Convertir las métricas a tipos estándar de Python
            processed_metricas = {key: (value.item() if hasattr(value, "item") else value) for key, value in metricas.items()}

            insert_dataset(file_name, processed_file_path, "NPS_Agente", description, list(charts.values()), metricas=json.dumps(processed_metricas))


            return render_template(
                'process_data.html',
                tipo_archivo="NPS_Agente",
                metricas={
                    "NPS Global": round(nps_final, 2),
                    "Promotores": metricas.get("Promotores", 0),
                    "Detractores": metricas.get("Detractores", 0),
                    "Pasivos": metricas.get("Pasivos", 0),
                    "Total de Respuestas": metricas.get("Total de Respuestas", 0),
                },
                **charts
            )

        else:
            flash("El archivo cargado no coincide con ningún formato esperado.", "error")
            return redirect(url_for('index'))

    except Exception as e:
        flash(f"Error al procesar los datos: {e}", "error")
        print(f"Error en /process_data: {e}")
        return redirect(url_for('index'))

@app.route('/list_datasets', methods=['GET'])
def list_datasets():
    """Vista para supervisores: lista de datasets procesados."""
    try:
        # Verifica que el usuario sea un Supervisor
        if session.get("puesto") != "Supervisor":
            flash("No tienes permiso para acceder a esta página.", "error")
            return redirect(url_for("index"))
        
        # Obtener datasets de la base de datos
        datasets = get_all_datasets()

        return render_template("list_datasets.html", datasets=datasets)

    except Exception as e:
        flash(f"Error al cargar los datasets: {e}", "error")
        print(f"Error en /list_datasets: {e}")
        return redirect(url_for("index"))
    
@app.route('/view_dataset/<int:dataset_id>', methods=['GET'])
def view_dataset(dataset_id):
    """Muestra los detalles de un dataset específico."""
    try:
        dataset = get_dataset_by_id(dataset_id)
        if not dataset:
            flash("No se encontró el dataset solicitado.", "error")
            return redirect(url_for('list_datasets'))

        # Verificar si metricas ya es un dict
        if isinstance(dataset.get('metricas'), dict):
            metricas = dataset['metricas']
        else:
            metricas = json.loads(dataset['metricas']) if dataset.get('metricas') else None

        # Procesar las rutas de los gráficos
        chart_paths = dataset.get("chart_paths", "").split(",")
        dataset["chart_paths"] = [
            path.strip() if path.startswith("/static/processed_charts/") else f"/static/processed_charts/{path.strip()}"
            for path in chart_paths
        ]

        # Verificar existencia de los archivos
        for chart in dataset["chart_paths"]:
            full_path = os.path.join(app.root_path, chart.lstrip("/"))
            if not os.path.exists(full_path):
                print(f"[WARNING] Gráfico no encontrado: {full_path}")

        # Renderizar el template
        return render_template('view_dataset.html', dataset=dataset, metricas=metricas)

    except Exception as e:
        flash(f"Error al mostrar el dataset: {e}", "error")
        print(f"[ERROR] Error en /view_dataset/{dataset_id}: {e}")
        return redirect(url_for('list_datasets'))



@app.route('/generate_analysis/<int:dataset_id>', methods=['GET'])
def generate_analysis(dataset_id):
    """Genera justificaciones automáticas usando LLM y crea un informe interactivo."""
    try:
        # Obtener el dataset
        dataset = get_dataset_by_id(dataset_id)

        if not dataset:
            flash("Dataset no encontrado.", "error")
            print("[ERROR] El dataset solicitado no existe en la base de datos.")
            return redirect(url_for('list_datasets'))

        # Extraer métricas
        metricas = dataset['metricas']
        if isinstance(metricas, str):
            metricas = json.loads(metricas)

        # Describir gráficos
        chart_descriptions = [
            "Gráfico " + path.split("/")[-1].replace("_", " ").capitalize()
            for path in dataset['chart_paths'].split(",")
        ]

        # Crear prompt para el modelo LLM
        prompt = f"""
        Genera un análisis detallado con base en las siguientes métricas y gráficos:
        {json.dumps(metricas, indent=2)}

        Gráficos asociados:
        {', '.join(chart_descriptions)}

        Tu objetivo:
        1. Proveer justificaciones automáticas basadas en las métricas.
        2. Comentar sobre posibles áreas de mejora.
        3. Redactar el informe como texto formal.
        """

        # Llamar al modelo LLM
        response = client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=[
                {"role": "system", "content": "Eres un asistente que redacta informes de análisis."},
                {"role": "user", "content": prompt}
            ]
        )

        # Acceder al contenido del mensaje de la respuesta
        analysis = response.choices[0].message.content  # Obtener el contenido generado
        
        """
        # Guardar el análisis en la sesión
        session[f"analysis_{dataset_id}"] = analysis
        """

        # **Guardar el análisis en la base de datos**
        update_analysis_in_db(dataset_id, analysis)

        flash("Análisis generado y guardado correctamente.", "success")

        # Renderizar el análisis en la web
        return render_template('generate_analysis.html', dataset=dataset, metricas=metricas, analysis=analysis)
        
    except Exception as e:
        # Registrar errores
        flash(f"Error al generar el análisis: {e}", "error")
        print(f"[ERROR] Error en /generate_analysis/{dataset_id}: {e}")
        return redirect(url_for('list_datasets'))

@app.route('/list_datasets_reporting', methods=['GET', 'POST'])
def list_datasets_reporting():
    """Muestra los datasets procesados con opción para limpiar los registros."""
    try:
        if request.method == 'POST':
            # Lógica para limpiar la base de datos
            conn = sqlite3.connect('./data/processed_datasets.db')
            cursor = conn.cursor()
            cursor.execute("DELETE FROM datasets")  # Elimina todos los registros
            conn.commit()
            conn.close()
            flash("Todos los datasets han sido eliminados.", "success")
            return redirect(url_for('list_datasets_reporting'))

        datasets = get_all_datasets()
        return render_template('list_datasets_reporting.html', datasets=datasets)
    
    except Exception as e:
        flash(f"Error al cargar los datasets: {e}", "error")
        print(f"Error en /list_datasets_reporting: {e}")
        return redirect(url_for('index'))

@app.route('/export_to_word/<int:dataset_id>', methods=['GET'])
def export_to_word(dataset_id):
    """Exporta el análisis del dataset a un archivo Word."""
    try:
        dataset = get_dataset_by_id(dataset_id)
        if not dataset:
            flash("Dataset no encontrado.", "error")
            return redirect(url_for('list_datasets'))

        metricas = dataset.get("metricas", {})
        analysis = session.get(f"analysis_{dataset_id}", "No se encontró el análisis.")  # Recuperar el análisis

        # Ruta absoluta del archivo Word
        base_dir = os.path.abspath(os.getcwd())
        output_path = os.path.join(base_dir, "informes", f"informe_{dataset['file_name'].replace('.csv', '')}.docx")

        # Exportar a Word
        export_to_word_simple(dataset, analysis, output_path)

        return send_file(output_path, as_attachment=True)
    
    except Exception as e:
        flash(f"Error al exportar el análisis: {e}", "error")
        print(f"[ERROR] Error en /export_to_word/{dataset_id}: {e}")
        return redirect(url_for('list_datasets'))


if __name__ == "__main__":
    app.run(debug=True)