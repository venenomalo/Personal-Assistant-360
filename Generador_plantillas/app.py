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