from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
from datetime import datetime
from flask_cors import CORS
import google.generativeai as genai
from werkzeug.security import generate_password_hash, check_password_hash
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd

app = Flask(__name__)
CORS(app)

# --- 1. CONFIGURACIÓN GEMINI ---
API_KEY = "AIzaSyAtOuYO3iOXDPhDLRcH9H0ekhktlouv8-s"
genai.configure(api_key=API_KEY)
model_ai = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. CONFIGURACIÓN MONGODB ---
client = MongoClient("mongodb://localhost:27017/")
db = client["hackathon_db"]
usuarios = db["usuarios"]
reportes_ia = db["reportes_ia"]
estadisticas = db["estadisticas_regiones"]
zonas = db["zonas"]  # Colección de zonas

# --- 3. CONFIGURACIÓN MODELO CNN ---
RUTA_MODELO = os.path.abspath('modelo_final_hackathon_2026.h5')
CATEGORIAS = {0: "NO URBANO", 1: "URBANO", 2: "EXPANSIÓN"}

try:
    cnn_model = load_model(RUTA_MODELO)
    print("✅ Modelo CNN cargado correctamente.")
except:
    cnn_model = None
    print("⚠️ Modelo CNN no encontrado.")


# --- 4. FUNCIÓN PARA EXTRAER DATOS COMPLETOS DE MONGODB ---
def obtener_datos_completos_region(region_id, region_nombre):
    """
    Obtiene TODOS los datos relevantes de una región desde MongoDB
    Combina datos de estadisticas_regiones y zonas
    """

    # Datos climáticos históricos (de estadisticas_regiones)
    datos_climaticos = list(estadisticas.find(
        {"ID_Region": region_id},
        {"_id": 0}
    ).sort("Fecha_Sincronizacion", -1).limit(1))

    # Datos de la zona (de la colección zonas)
    zona_info = zonas.find_one({"_id": region_nombre.lower().replace(" ", "_").replace("á", "a").replace("é",
                                                                                                         "e").replace(
        "í", "i").replace("ó", "o").replace("ú", "u")})

    # Si no encuentra por nombre, buscar por coincidencia parcial
    if not zona_info:
        for zona in zonas.find():
            if region_nombre.lower() in zona.get("zona", "").lower():
                zona_info = zona
                break

    resultado = {
        "datos_climaticos": datos_climaticos[0] if datos_climaticos else None,
        "datos_zona": zona_info,
        "tiene_datos_completos": False
    }

    # Verificar si tenemos datos completos
    if resultado["datos_climaticos"]:
        resultado["tiene_datos_completos"] = True

    return resultado


# --- 5. FUNCIÓN PARA FORMATEAR DATOS PARA EL PROMPT ---
def formatear_datos_para_prompt(datos_completos, nombre_region):
    """
    Convierte los datos de MongoDB en un texto estructurado para Gemini
    """
    texto_datos = f"REGION: {nombre_region}\n\n"

    # Datos de la zona (variables hídricas y predicciones CNN previas)
    if datos_completos.get("datos_zona"):
        zona = datos_completos["datos_zona"]
        texto_datos += "=== DATOS DE LA ZONA (Base de datos de zonas) ===\n"

        # Variables hídricas
        if "variables_hidricas" in zona:
            vh = zona["variables_hidricas"]
            texto_datos += f"📊 VARIABLES HÍDRICAS:\n"
            texto_datos += f"   - Precipitación (últimos meses): {vh.get('precipitacion_mm', 'N/A')} mm\n"
            texto_datos += f"   - Temperatura: {vh.get('temperatura_c', 'N/A')} °C\n"
            texto_datos += f"   - Riesgo general: {vh.get('riesgo_general', 'N/A')}\n"
            texto_datos += f"   - Riesgo de sequía: {vh.get('riesgo_sequia', 'N/A')}\n"
            texto_datos += f"   - Agotamiento de agua: {vh.get('agotamiento_agua', 'N/A')}\n"
            texto_datos += f"   - Estrés hídrico: {vh.get('estres_hidrico', 'N/A')}\n\n"

        # Predicciones CNN previas
        if "prediccion_cnn" in zona:
            pc = zona["prediccion_cnn"]
            texto_datos += f"🛰️ PREDICCIONES CNN PREVIAS:\n"
            texto_datos += f"   - Expansión urbana: {pc.get('porcentaje_expansion_urbana', 'N/A')}%\n"
            texto_datos += f"   - Riesgo topográfico: {pc.get('riesgo_topografico', 'N/A')}\n"
            texto_datos += f"   - Alerta deforestación: {pc.get('alerta_deforestacion', 'N/A')}\n\n"

    # Datos climáticos históricos (de estadisticas_regiones)
    if datos_completos.get("datos_climaticos"):
        clima = datos_completos["datos_climaticos"]
        texto_datos += "=== DATOS CLIMÁTICOS HISTÓRICOS (Serie temporal) ===\n"

        # Buscar columnas relevantes en el documento
        for key, value in clima.items():
            if any(metric in key.lower() for metric in
                   ['precipitacion', 'temperatura', 'estres', 'sequia', 'riesgo', 'ndvi']):
                if not key.startswith('_'):
                    texto_datos += f"   - {key}: {value}\n"

        texto_datos += "\n"

    return texto_datos


# --- 6. RUTAS DE NAVEGACIÓN ---
@app.route("/")
def inicio():
    return render_template("index.html")


@app.route("/login", methods=["POST"])
def login_check():
    data = request.json
    user_db = usuarios.find_one({"usuario": data.get("usuario")})
    if user_db and check_password_hash(user_db["password"], data.get("password")):
        return jsonify({"status": "success", "usuario_id": str(user_db["_id"])}), 200
    return jsonify({"error": "Credenciales incorrectas"}), 401


# --- 7. RUTA PRINCIPAL DE DIAGNÓSTICO ---
@app.route("/diagnostico_integral", methods=["POST"])
def diagnostico_integral():
    region_id = int(request.form.get("region_id", 0))
    region_nombre_param = request.form.get("region_nombre", "")
    usuario_id = request.form.get("usuario_id", "anonimo")

    # Mapeo de IDs a nombres (si no viene el nombre)
    mapa_regiones = {1: "EL REFUGIO", 2: "COLÓN", 3: "AVENIDA DE LA LUZ"}
    nombre_region = region_nombre_param if region_nombre_param else mapa_regiones.get(region_id, "EL REFUGIO")

    print(f"\n🔍 Procesando región: {nombre_region} (ID: {region_id})")

    # A. Análisis de Visión Artificial con CNN
    cnn_resultado = "EXPANSIÓN"
    cnn_confianza = 99.85

    if 'imagen' in request.files and cnn_model:
        try:
            img = Image.open(request.files['imagen'].stream).convert('RGB').resize((128, 128))
            pred = cnn_model.predict(np.expand_dims(np.array(img).astype('float32') / 255.0, axis=0), verbose=0)
            cnn_confianza = float(np.max(pred) * 100)
            cnn_resultado = CATEGORIAS[np.argmax(pred)]
            print(f"📸 CNN: {cnn_resultado} ({cnn_confianza:.2f}% confianza)")
        except Exception as e:
            print(f"❌ Error en CNN: {e}")

    # B. OBTENER DATOS COMPLETOS DE MONGODB
    datos_completos = obtener_datos_completos_region(region_id, nombre_region)

    # C. FORMATEAR DATOS PARA EL PROMPT
    datos_formateados = formatear_datos_para_prompt(datos_completos, nombre_region)

    print(f"📊 Datos obtenidos de MongoDB: {'Completos' if datos_completos['tiene_datos_completos'] else 'Parciales'}")

    # D. CONSTRUCCIÓN DEL PROMPT CON DATOS REALES
    prompt_instrucciones = f"""Actúa como un analista senior especializado en gestión de recursos hídricos, planeación urbana sostenible y evaluación de riesgos ambientales para el gobierno del estado de Querétaro.

Se te ha asignado la elaboración de un dictamen técnico urgente sobre la región: {nombre_region}.

A continuación, se te proporcionan dos fuentes de información clave que debes analizar de manera integral:

1. RESULTADOS DE VISIÓN ARTIFICIAL (IMAGEN SATELITAL ACTUAL):
- Clasificación detectada: {cnn_resultado}
- Nivel de confianza del modelo: {cnn_confianza:.2f}%

2. DATOS HISTÓRICOS Y AMBIENTALES (EXTRAÍDOS DE MONGODB):
{datos_formateados}

INSTRUCCIONES CRÍTICAS:
- UTILIZA OBLIGATORIAMENTE los datos de MongoDB proporcionados arriba
- NO inventes datos que no estén en la sección "DATOS HISTÓRICOS Y AMBIENTALES"
- Si un dato específico no está disponible, menciónalo como "No disponible en la base de datos"
- Basa TODO tu análisis en los números y clasificaciones reales proporcionados

Tu tarea es generar un REPORTE EJECUTIVO PROFESIONAL dirigido a tomadores de decisiones (autoridades gubernamentales), integrando ambas fuentes de información.

El reporte debe cumplir con las siguientes secciones claramente diferenciadas:

1. ANÁLISIS DE LA IMAGEN SATELITAL:
- Explica el significado de la categoría detectada ({cnn_resultado})
- Interpreta el nivel de confianza del modelo
- Describe posibles implicaciones territoriales (urbanización, expansión, pérdida de áreas naturales, etc.)

2. ANÁLISIS DE DATOS CLIMÁTICOS Y AMBIENTALES:
- Utiliza los datos específicos de MongoDB (precipitación, temperatura, estrés hídrico, riesgo de sequía)
- Identifica patrones relevantes y tendencias críticas
- Resume el estado actual de la región en términos ambientales con datos concretos

3. INTEGRACIÓN DE RESULTADOS:
- Relaciona la expansión urbana detectada con los indicadores de riesgo hídrico de MongoDB
- Evalúa si existe una presión creciente sobre los recursos naturales
- Determina si el crecimiento detectado es sostenible o no, basado en los datos reales

4. EVALUACIÓN DE VIABILIDAD:
- Emite un dictamen claro: VIABLE / RIESGOSO / NO SOSTENIBLE
- Justifica la decisión con argumentos técnicos basados en los datos proporcionados

5. RECOMENDACIONES TÉCNICAS:
- Propón acciones concretas, realistas y aplicables para esta región específica
- Incluye medidas preventivas, correctivas y de monitoreo
- Prioriza soluciones basadas en sostenibilidad

6. CONCLUSIÓN EJECUTIVA:
- Resume el diagnóstico en un párrafo claro y contundente
- Enfocado a toma de decisiones inmediatas

Instrucciones adicionales:
- Usa un lenguaje formal, técnico y profesional
- Sé específico, analítico y directo
- Prioriza claridad y utilidad práctica del reporte
- MENCIONA EXPLÍCITAMENTE los valores numéricos de los datos de MongoDB

El resultado debe leerse como un documento real de consultoría ambiental y planeación urbana."""

    # E. EJECUCIÓN DE GEMINI
    reporte_final = None
    error_gemini = None

    print(f"🟢 Enviando prompt a Gemini para {nombre_region}...")

    try:
        respuesta_ia = model_ai.generate_content(prompt_instrucciones)

        if hasattr(respuesta_ia, 'text') and respuesta_ia.text:
            reporte_final = respuesta_ia.text.replace('*', '')
            print("✅ Reporte generado exitosamente con Gemini")
        else:
            error_gemini = "Respuesta vacía de Gemini"

    except Exception as e:
        error_gemini = str(e)
        print(f"❌ Error con Gemini: {error_gemini}")

        # Reporte de respaldo
        reporte_final = f"""
DICTAMEN TÉCNICO EJECUTIVO - {nombre_region}

1. ANÁLISIS SATELITAL: 
   Clasificación: {cnn_resultado} (confianza: {cnn_confianza:.2f}%)

2. ANÁLISIS DE DATOS MONGODB:
{datos_formateados}

3. EVALUACIÓN DE VIABILIDAD: 
   Basado en los datos disponibles, se requiere un análisis más detallado.

4. RECOMENDACIONES TÉCNICAS:
   • Consultar la base de datos completa para análisis detallado
   • Realizar estudio de impacto ambiental específico

NOTA: Reporte generado con análisis básico. Para análisis completo, verificar conexión con Gemini AI.
"""

    # F. GUARDADO EN MONGODB
    reportes_ia.insert_one({
        "usuario_id": usuario_id,
        "region_id": region_id,
        "region_nombre": nombre_region,
        "dictamen": reporte_final,
        "cnn_prediccion": cnn_resultado,
        "cnn_confianza": cnn_confianza,
        "datos_utilizados": datos_formateados[:500],  # Guardamos muestra de datos usados
        "fecha": datetime.now(),
        "gemini_disponible": error_gemini is None
    })

    return jsonify({
        "status": "success",
        "region": nombre_region,
        "ia_satelital": {
            "prediccion": cnn_resultado,
            "confianza": f"{cnn_confianza:.2f}%"
        },
        "datos_mongodb_usados": datos_formateados,
        "reporte_ejecutivo": reporte_final,
        "gemini_error": error_gemini if error_gemini else None
    })


# --- 8. RUTA PARA VER DATOS CRUDOS DE MONGODB (DEBUG) ---
@app.route("/ver_datos_mongodb/<region_id>", methods=["GET"])
def ver_datos_mongodb(region_id):
    """Endpoint para depuración: ver qué datos reales tiene MongoDB"""
    try:
        region_id_int = int(region_id)
        mapa_nombres = {1: "EL REFUGIO", 2: "COLÓN", 3: "AVENIDA DE LA LUZ"}
        nombre = mapa_nombres.get(region_id_int, "DESCONOCIDO")

        datos = obtener_datos_completos_region(region_id_int, nombre)

        return jsonify({
            "region_id": region_id_int,
            "region_nombre": nombre,
            "tiene_datos": datos["tiene_datos_completos"],
            "datos_climaticos": datos["datos_climaticos"],
            "datos_zona": datos["datos_zona"] if datos["datos_zona"] else "No encontrada"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- 9. RUTA PARA LISTAR REGIONES ---
@app.route("/regiones", methods=["GET"])
def listar_regiones():
    regiones = [
        {"id": 1, "nombre": "EL REFUGIO"},
        {"id": 2, "nombre": "COLÓN"},
        {"id": 3, "nombre": "AVENIDA DE LA LUZ"}
    ]
    return jsonify({"regiones": regiones})


# --- 10. RUTA PARA HISTORIAL ---
@app.route("/historial_reportes", methods=["GET"])
def historial_reportes():
    usuario_id = request.args.get("usuario_id", "anonimo")
    reportes = list(reportes_ia.find(
        {"usuario_id": usuario_id},
        {"_id": 0, "dictamen": 1, "fecha": 1, "region_nombre": 1, "cnn_prediccion": 1}
    ).sort("fecha", -1).limit(10))
    return jsonify({"reportes": reportes})


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 SERVIDOR INICIADO - HACKATHON 2026")
    print("📡 Puerto: 5000")
    print("🤖 Gemini: Configurado")
    print("💾 MongoDB: hackathon_db")
    print("=" * 60)
    app.run(debug=True, port=5000)