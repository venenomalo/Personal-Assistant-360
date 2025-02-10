import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os
from Generador_plantillas.db_utils import save_chart

CHARTS_FOLDER = os.path.join(os.getcwd(), 'Generador_plantillas', 'static', 'processed_charts')

# Verifica si la carpeta existe antes de crearla
if not os.path.exists(CHARTS_FOLDER):
    os.makedirs(CHARTS_FOLDER, exist_ok=True)

def process_nds_servicio(data):
    """Procesa los datos para archivos de tipo NDS_Servicio y genera gráficos."""
    # Convertir 'Waiting Time (AVG)' y 'Talk Duration (AVG)' a segundos
    for column in ['Waiting Time (AVG)', 'Talk Duration (AVG)']:
        if column in data.columns:
            data[column] = pd.to_timedelta(data[column]).dt.total_seconds()

    # Convertir 'Service Level (20 Seconds)' a porcentaje
    data['Service Level (20 Seconds)'] = (
        data['Service Level (20 Seconds)']
        .str.replace('%', '', regex=False)
        .astype(float) / 100
    )

    # Manejar valores nulos
    data.fillna(0, inplace=True)

    # Cálculo del NDS del servicio
    data['Weighted_Service_Level'] = data['Service Level (20 Seconds)'] * data['Incoming Calls']
    total_incoming_calls = data['Incoming Calls'].sum()
    nds_servicio = data['Weighted_Service_Level'].sum() / total_incoming_calls if total_incoming_calls > 0 else 0

    metricas = {
        "NDS del Servicio": nds_servicio,
        "Llamadas Entrantes": total_incoming_calls,
        "Llamadas Contestadas": data["Answered Calls"].sum(),
        "Llamadas Abandonadas": data["Abandoned Calls"].sum(),
    }

    # Guardar los datos procesados
    processed_file_path = "./data/processed_files/NDS_Servicio_Procesado.csv"
    os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
    data.to_csv(processed_file_path, index=False)

    # Generar gráficos
    charts = {}

    # 1. Gráfico de Barras: Llamadas Contestadas vs Abandonadas
    if 'Answered Calls' in data.columns and 'Abandoned Calls' in data.columns:
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        data[['Answered Calls', 'Abandoned Calls']].sum().plot(kind='bar', ax=ax1, color=['green', 'red'])
        ax1.tick_params(axis='x', rotation=0)
        ax1.set_title('Llamadas Contestadas vs Abandonadas')
        ax1.set_ylabel('Número de Llamadas')
        charts['calls_chart'] = save_chart(fig1, 'nds_calls_chart.png')

    # 2. Gráfico de Línea: Evolución del Nivel de Servicio
    if 'Index' in data.columns:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        data.plot(x='Index', y='Service Level (20 Seconds)', kind='line', ax=ax2, color='blue', linewidth=2)
        ax2.set_xlabel('Índice')
        ax2.set_ylabel('Nivel de Servicio (%)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_title('Evolución del Nivel de Servicio')
        charts['service_level_chart'] = save_chart(fig2, 'nds_service_level_chart.png')

    # 3. Gráfico de Dispersión: Nivel de Servicio vs Tiempo de Espera
    if 'Waiting Time (AVG)' in data.columns:
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        data.plot.scatter(x='Service Level (20 Seconds)', y='Waiting Time (AVG)', ax=ax3, color='purple')
        ax3.set_xlabel('Nivel de Servicio (%)')
        ax3.set_ylabel('Tiempo de Espera Promedio (segundos)')
        ax3.set_title('Nivel de Servicio vs Tiempo de Espera')
        charts['service_vs_waiting_chart'] = save_chart(fig3, 'nds_service_vs_waiting_chart.png')

    # 4. Gráfico de Barras: Talk Duration (AVG)
    if 'Talk Duration (AVG)' in data.columns:
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        data['Talk Duration (AVG)'].plot(kind='hist', bins=20, ax=ax4, color='orange', alpha=0.7)
        ax4.set_xlabel('Duración (segundos)')
        ax4.set_ylabel('Frecuencia')
        ax4.set_title('Distribución de la Duración Promedio de Conversación')
        charts['talk_duration_chart'] = save_chart(fig4, 'nds_talk_duration_chart.png')

    return nds_servicio, charts, data, metricas

def process_nps_agente(data):
    """Procesa los datos para archivos de tipo NPS_Agente y genera gráficos."""
    # Filtrar filas con respuestas válidas
    data_valid = data[data['Answered (Y/N)'] == 'Y'].copy()

    # Convertir Satisfaction rating a escala 0-10
    data_valid['Satisfaction rating'] = data_valid['Satisfaction rating'] * 2

    # Clasificar como Promotores, Detractores y Pasivos
    data_valid['NPS'] = data_valid['Satisfaction rating'].apply(
        lambda x: 100 if x >= 9 else (-100 if x <= 6 else 0)
    )
    # Asegurar columnas numéricas
    if 'Speed of answer in seconds' in data_valid.columns:
        data_valid['Speed of answer in seconds'] = pd.to_numeric(
            data_valid['Speed of answer in seconds'], errors='coerce'
        ).fillna(0)

    if 'AvgTalkDuration' in data_valid.columns:
        data_valid['AvgTalkDuration'] = pd.to_timedelta(
            data_valid['AvgTalkDuration'], errors='coerce'
        ).dt.total_seconds().fillna(0)

    # Calcular métricas
    nps_promoters = len(data_valid[data_valid['NPS'] == 100])
    nps_detractors = len(data_valid[data_valid['NPS'] == -100])
    nps_passives = len(data_valid[(data_valid['NPS'] != 100) & (data_valid['NPS'] != -100)])
    total_responses = len(data_valid)

    nps_final = ((nps_promoters - nps_detractors) / total_responses) * 100 if total_responses > 0 else 0

    metricas = {
        "NPS Global": nps_final,
        "Promotores": nps_promoters,
        "Detractores": nps_detractors,
        "Pasivos": nps_passives,
        "Total de Respuestas": total_responses,
    }
    
    # Guardar los datos procesados
    processed_file_path = "./data/processed_files/NPS_Agente_Procesado.csv"
    os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
    data_valid.to_csv(processed_file_path, index=False)

    # Generar gráficos
    charts = {}  
  
    # 1. Gráfico de Pastel: Distribución de Promotores, Pasivos y Detractores
    if 'Topic' in data_valid.columns:
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        labels = ['Promotores', 'Pasivos', 'Detractores']
        sizes = [nps_promoters, nps_passives, nps_detractors]
        colors = ['green', 'yellow', 'red']
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax1.axis('equal')
        charts['pie_chart'] = save_chart(fig1, 'nps_pie_chart.png')

    # 2. Gráfico de Barras: NPS por Tema (Topic)
    if 'Topic' in data_valid.columns:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        data_valid.groupby('Topic')['NPS'].mean().plot(kind='bar', ax=ax2, color='skyblue')
        ax2.set_ylabel('NPS')
        ax2.set_xlabel('Tema')
        ax2.tick_params(axis='x', rotation=0)
        charts['topic_chart'] = save_chart(fig2, 'nps_topic_chart.png')

    # 3. Gráfico de Barras: Velocidad de Respuesta por Categoría de NPS
    if 'Speed of answer in seconds' in data_valid.columns:
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        data_valid.groupby('NPS')['Speed of answer in seconds'].mean().plot(kind='bar', ax=ax3, color='orange')
        ax3.set_xlabel('Categoría de NPS')
        ax3.set_ylabel('Velocidad de Respuesta (segundos)')
        ax3.tick_params(axis='x', rotation=0)
        charts['speed_chart'] = save_chart(fig3, 'nps_speed_chart.png')

    # 4. Gráfico de Barras: Duración Promedio de Conversación por Categoría de NPS
    if 'AvgTalkDuration' in data_valid.columns:
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        data_valid.groupby('NPS')['AvgTalkDuration'].mean().plot(kind='bar', ax=ax4, color='purple')
        ax4.set_xlabel('Categoría de NPS')
        ax4.set_ylabel('Duración Promedio (segundos)')
        ax4.tick_params(axis='x', rotation=0)
        charts['duration_chart'] = save_chart(fig4, 'nps_duration_chart.png')

    # 5. Gráfico de Barras: NPS por Agente
    if 'Agent' in data_valid.columns:
        fig5, ax5 = plt.subplots(figsize=(8, 6))
        data_valid.groupby('Agent')['NPS'].mean().plot(kind='bar', ax=ax5, color='blue')
        ax5.set_ylabel('NPS')
        ax5.set_xlabel('Agente')
        ax5.tick_params(axis='x', rotation=0)
        charts['agent_chart'] = save_chart(fig5, 'nps_agent_chart.png')

    return nps_final, charts, data_valid, metricas
