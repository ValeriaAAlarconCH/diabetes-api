from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import traceback
import os

# ======================= CONFIGURACIÓN =======================
app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

# Variables globales para el modelo y configuración
modelo = None
config = None
FEATURE_NAMES = []
TARGET_NAMES = []
CATEGORICAL_MAPPING = {}

# ======================= CARGA DEL MODELO =======================
def cargar_modelo_y_config():
    """Carga el modelo y la configuración desde archivos .pkl"""
    global modelo, config, FEATURE_NAMES, TARGET_NAMES, CATEGORICAL_MAPPING
    
    try:
        print("🚀 Cargando configuración del modelo...")
        
        # 1. Cargar configuración
        config_path = 'model_config.pkl'
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")
        
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        # Extraer información importante
        FEATURE_NAMES = config.get('feature_names', [])
        TARGET_NAMES = config.get('target_names', [])
        CATEGORICAL_MAPPING = config.get('categorical_mapping', {})
        
        print(f"✅ Configuración cargada: {len(FEATURE_NAMES)} features, {len(TARGET_NAMES)} clases")
        
        # 2. Cargar modelo
        model_path = 'modelo_diabetes_corregido.pkl'
        if not os.path.exists(model_path):
            # Intentar con el nombre original
            model_path = 'modelo_diabetes.pkl'
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Archivo del modelo no encontrado: {model_path}")
        
        with open(model_path, 'rb') as f:
            modelo = pickle.load(f)
        
        print(f"✅ Modelo cargado desde: {model_path}")
        print(f"📊 Tipo de modelo: {type(modelo).__name__}")
        
        # 3. Verificar y limpiar atributos problemáticos del modelo
        if hasattr(modelo, 'use_label_encoder'):
            delattr(modelo, 'use_label_encoder')
            print("⚠️  Atributo 'use_label_encoder' eliminado")
        
        # Eliminar otros atributos problemáticos
        problematic_attrs = ['_le', 'label_encoder', 'classes_']
        for attr in problematic_attrs:
            if hasattr(modelo, attr):
                delattr(modelo, attr)
        
        return True
        
    except Exception as e:
        print(f"❌ Error crítico al cargar el modelo: {str(e)}")
        traceback.print_exc()
        return False

# ======================= PREPARACIÓN DE DATOS =======================
def preparar_datos_para_modelo(datos_json):
    """
    Convierte los datos JSON recibidos en un formato compatible con el modelo.
    
    Args:
        datos_json: Diccionario con los datos del paciente
        
    Returns:
        np.array: Array 2D con las características preparadas
    """
    # Crear array inicial con ceros
    features = np.zeros(len(FEATURE_NAMES))
    
    # Mapeo simplificado de nombres (español -> feature_name)
    mapeo_nombres = {
        # Variables numéricas
        'edad': 'edad',
        'niveles_glucosa': 'niveles_glucosa',
        'niveles_insulina': 'niveles_insulina',
        'indice_masa_corporal': 'indice_masa_corporal',
        'presion_arterial': 'presion_arterial',
        'niveles_colesterol': 'niveles_colesterol',
        'circunferencia_cintura': 'circunferencia_cintura',
        'salud_pancreatica': 'salud_pancreatica',
        'funcion_pulmonar': 'funcion_pulmonar',
        'evaluaciones_neurologicas': 'evaluaciones_neurologicas',
        'niveles_enzimas_digestivas': 'niveles_enzimas_digestivas',
        'peso_nacimiento': 'peso_nacimiento',
        'aumento_peso_embarazo': 'aumento_peso_embarazo',
        
        # Variables categóricas
        'autoanticuerpos': 'autoanticuerpos',
        'antecedentes_familiares': 'antecedentes_familiares',
        'marcadores_geneticos': 'marcadores_geneticos',
        'actividad_fisica': 'actividad_fisica',
        'estado_tabaquismo': 'estado_tabaquismo',
        'consumo_alcohol': 'consumo_alcohol',
        'prueba_orina': 'prueba_orina',
        'etnicidad': 'etnicidad',
        'habitos_alimenticios': 'habitos_alimenticios',
        'prueba_tolerancia_glucosa': 'prueba_tolerancia_glucosa',
        'factores_ambientales': 'factores_ambientales',
        'uso_esteroides': 'uso_esteroides',
        'diagnostico_fibrosis_quistica': 'diagnostico_fibrosis_quistica',
        'historial_pcos': 'historial_pcos',
        'diabetes_gestacional_previa': 'diabetes_gestacional_previa',
        'historial_embarazos': 'historial_embarazos',
        'pruebas_geneticas': 'pruebas_geneticas',
        'pruebas_funcion_hepatica': 'pruebas_funcion_hepatica',
        'sintomas_inicio_temprano': 'sintomas_inicio_temprano',
        'factores_socioeconomicos': 'factores_socioeconomicos'
    }
    
    # Procesar cada feature esperado
    for i, feature_name in enumerate(FEATURE_NAMES):
        valor = None
        
        # Buscar el valor usando diferentes estrategias
        if feature_name in datos_json:
            valor = datos_json[feature_name]
        else:
            # Buscar por mapeo
            for key_espanol, key_ingles in mapeo_nombres.items():
                if key_ingles == feature_name and key_espanol in datos_json:
                    valor = datos_json[key_espanol]
                    break
        
        # Si encontramos un valor, procesarlo
        if valor is not None:
            try:
                # Verificar si es variable categórica
                if feature_name in CATEGORICAL_MAPPING:
                    # Convertir valor categórico a numérico
                    valor_str = str(valor).strip()
                    if valor_str in CATEGORICAL_MAPPING[feature_name]:
                        features[i] = CATEGORICAL_MAPPING[feature_name][valor_str]
                    else:
                        # Valor por defecto para categorías desconocidas
                        features[i] = 0.0
                else:
                    # Variable numérica
                    features[i] = float(valor)
            except (ValueError, TypeError):
                features[i] = 0.0
    
    return features.reshape(1, -1)

# ======================= PREDICCIÓN DE EMERGENCIA =======================
def prediccion_emergencia(datos_json):
    """
    Predicción simplificada para cuando el modelo principal falla.
    Basada en reglas clínicas básicas.
    """
    # Extraer valores clave
    edad = float(datos_json.get('edad', 45))
    glucosa = float(datos_json.get('niveles_glucosa', 100))
    insulina = float(datos_json.get('niveles_insulina', 15))
    autoanticuerpos = str(datos_json.get('autoanticuerpos', 'Negative')).strip()
    
    # Reglas de decisión
    if glucosa > 200 or (glucosa > 150 and autoanticuerpos == 'Positive'):
        clase_predicha = 'Type 1 Diabetes'
        probabilidad = 0.85
    elif glucosa > 126:
        clase_predicha = 'Type 2 Diabetes'
        probabilidad = 0.75
    elif glucosa >= 100:
        clase_predicha = 'Prediabetic'
        probabilidad = 0.65
    elif 'gestacional' in str(datos_json.get('historial_embarazos', '')).lower():
        clase_predicha = 'Gestational Diabetes'
        probabilidad = 0.70
    else:
        clase_predicha = 'Type 2 Diabetes'  # Default más común
        probabilidad = 0.60
    
    # Generar probabilidades para todas las clases
    probabilidades = {}
    for clase in TARGET_NAMES:
        if clase == clase_predicha:
            probabilidades[clase] = probabilidad
        else:
            probabilidades[clase] = (1 - probabilidad) / (len(TARGET_NAMES) - 1)
    
    return {
        'predictedClass': clase_predicha,
        'probability': probabilidad,
        'probabilities': probabilidades,
        'featureImportance': {},
        'success': True,
        'message': 'Predicción usando reglas clínicas (modelo temporalmente no disponible)'
    }

# ======================= ENDPOINTS DE LA API =======================
@app.route('/health', methods=['GET'])
def verificar_salud():
    """Verifica que el servicio esté funcionando correctamente"""
    if modelo is None or config is None:
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False,
            'message': 'Modelo no cargado'
        }), 500
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'model_type': type(modelo).__name__,
        'num_features': len(FEATURE_NAMES),
        'num_classes': len(TARGET_NAMES),
        'classes': TARGET_NAMES,
        'timestamp': os.times().user  # Tiempo de CPU del proceso
    })

@app.route('/predict', methods=['POST'])
def predecir():
    """Endpoint principal para realizar predicciones"""
    try:
        # Verificar que el modelo esté cargado
        if modelo is None:
            return jsonify({
                'error': 'Modelo no disponible',
                'success': False,
                'message': 'El servicio de predicción no está listo'
            }), 503
        
        # Obtener y validar datos
        datos = request.json
        if not datos:
            return jsonify({
                'error': 'Datos no proporcionados',
                'success': False
            }), 400
        
        print(f"📥 Recibida solicitud de predicción con {len(datos)} campos")
        
        # Preparar datos para el modelo
        datos_preparados = preparar_datos_para_modelo(datos)
        
        # Realizar predicción
        try:
            prediccion = modelo.predict(datos_preparados)[0]
            probabilidades = modelo.predict_proba(datos_preparados)[0]
        except Exception as e:
            print(f"⚠️  Error en predicción del modelo: {e}")
            print("🔄 Usando predicción de emergencia...")
            return jsonify(prediccion_emergencia(datos))
        
        # Validar índice de predicción
        if prediccion >= len(TARGET_NAMES):
            print(f"⚠️  Índice de predicción inválido: {prediccion}")
            return jsonify(prediccion_emergencia(datos))
        
        # Construir respuesta
        clase_predicha = TARGET_NAMES[prediccion]
        probabilidad_principal = float(probabilidades[prediccion])
        
        # Calcular todas las probabilidades
        todas_probabilidades = {}
        for i, clase in enumerate(TARGET_NAMES):
            if i < len(probabilidades):
                todas_probabilidades[clase] = float(probabilidades[i])
        
        # Calcular importancia de características si está disponible
        importancia = {}
        if hasattr(modelo, 'feature_importances_'):
            for i, feature in enumerate(FEATURE_NAMES):
                if i < len(modelo.feature_importances_):
                    importancia[feature] = float(modelo.feature_importances_[i])
        
        respuesta = {
            'predictedClass': clase_predicha,
            'probability': probabilidad_principal,
            'probabilities': todas_probabilidades,
            'featureImportance': importancia,
            'success': True,
            'message': 'Predicción exitosa',
            'modelUsed': type(modelo).__name__
        }
        
        print(f"✅ Predicción exitosa: {clase_predicha} ({probabilidad_principal:.1%})")
        return jsonify(respuesta)
        
    except Exception as e:
        print(f"❌ Error interno en predicción: {str(e)}")
        traceback.print_exc()
        
        return jsonify({
            'error': 'Error interno del servidor',
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/features', methods=['GET'])
def obtener_caracteristicas():
    """Obtiene información sobre las características del modelo"""
    if config is None:
        return jsonify({'error': 'Configuración no cargada'}), 500
    
    # Preparar información de características categóricas
    cat_info = {}
    for feature, mapping in CATEGORICAL_MAPPING.items():
        cat_info[feature] = {
            'type': 'categorical',
            'values': list(mapping.keys()),
            'mapping': mapping
        }
    
    return jsonify({
        'num_features': len(FEATURE_NAMES),
        'num_classes': len(TARGET_NAMES),
        'features': FEATURE_NAMES,
        'classes': TARGET_NAMES,
        'categorical_features': cat_info,
        'note': 'Para predicción, envíe datos JSON con estos nombres de características'
    })

@app.route('/test', methods=['GET', 'POST'])
def prueba_api():
    """Endpoint de prueba con datos de ejemplo"""
    
    # Datos de ejemplo realistas
    datos_ejemplo = {
        'edad': 45,
        'niveles_glucosa': 180.0,
        'niveles_insulina': 35.0,
        'indice_masa_corporal': 28.5,
        'autoanticuerpos': 'Negative',
        'antecedentes_familiares': 'Yes',
        'marcadores_geneticos': 'Positive',
        'presion_arterial': 130.0,
        'niveles_colesterol': 220.0,
        'circunferencia_cintura': 95.0,
        'actividad_fisica': 'Low',
        'estado_tabaquismo': 'Non-Smoker',
        'consumo_alcohol': 'Moderate',
        'prueba_orina': 'Normal',
        'etnicidad': 'High Risk',
        'habitos_alimenticios': 'Unhealthy',
        'prueba_tolerancia_glucosa': 'Abnormal',
        'factores_ambientales': 'Present',
        'uso_esteroides': 'No',
        'diagnostico_fibrosis_quistica': 'No',
        'historial_pcos': 'No',
        'diabetes_gestacional_previa': 'No',
        'historial_embarazos': 'Normal',
        'pruebas_geneticas': 'Negative',
        'pruebas_funcion_hepatica': 'Normal',
        'sintomas_inicio_temprano': 'No',
        'factores_socioeconomicos': 'Medium'
    }
    
    if request.method == 'POST':
        # Realizar predicción con los datos de ejemplo
        try:
            # Usar la función de predicción directamente
            request._json = datos_ejemplo
            return predecir()
        except Exception as e:
            return jsonify({
                'error': str(e),
                'success': False,
                'test_data': datos_ejemplo
            }), 500
    else:
        # Método GET: Mostrar información
        return jsonify({
            'message': 'Para probar la API, envíe un POST a este endpoint o use POST /predict',
            'example_request': {
                'method': 'POST',
                'url': '/test',
                'content_type': 'application/json',
                'data': datos_ejemplo
            },
            'available_endpoints': {
                'GET /health': 'Verificar estado del servicio',
                'GET /features': 'Obtener características del modelo',
                'POST /predict': 'Realizar predicción (envíe datos JSON)',
                'GET/POST /test': 'Probar con datos de ejemplo'
            }
        })

@app.route('/config', methods=['GET'])
def obtener_configuracion():
    """Obtiene la configuración completa del modelo (para debugging)"""
    if config is None:
        return jsonify({'error': 'Configuración no cargada'}), 500
    
    return jsonify({
        'feature_names': FEATURE_NAMES,
        'target_names': TARGET_NAMES,
        'categorical_mapping': CATEGORICAL_MAPPING,
        'model_info': {
            'type': str(type(modelo)),
            'has_feature_importance': hasattr(modelo, 'feature_importances_')
        }
    })

# ======================= INICIALIZACIÓN =======================
def iniciar_servidor():
    """Función principal para iniciar el servidor"""
    print("=" * 60)
    print("   🩺 API DE PREDICCIÓN DE DIABETES - ML")
    print("=" * 60)
    
    # Cargar modelo
    if not cargar_modelo_y_config():
        print("\n❌ NO SE PUDO INICIAR EL SERVICIO")
        print("\n🔧 POSIBLES SOLUCIONES:")
        print("   1. Verifica que los archivos estén en la misma carpeta:")
        print("      - model_config.pkl")
        print("      - modelo_diabetes_corregido.pkl (o modelo_diabetes.pkl)")
        print("   2. Ejecuta fix_model.py si hay problemas con el modelo")
        print("   3. Verifica permisos de los archivos")
        return False
    
    print("\n✅ SERVICIO INICIADO CORRECTAMENTE")
    print("\n📡 ENDPOINTS DISPONIBLES:")
    print("   GET  /health     → Estado del servicio")
    print("   GET  /features   → Información del modelo")
    print("   POST /predict    → Realizar predicción")
    print("   GET/POST /test   → Probar con ejemplo")
    print("   GET  /config     → Configuración (debug)")
    
    print("\n🌐 URL PRINCIPAL: http://localhost:5000")
    print("📊 INFORMACIÓN DEL MODELO:")
    print(f"   • Features: {len(FEATURE_NAMES)}")
    print(f"   • Clases: {len(TARGET_NAMES)}")
    print(f"   • Tipo: {type(modelo).__name__}")
    
    print("\n" + "=" * 60)
    return True

# ======================= EJECUCIÓN PRINCIPAL =======================
if __name__ == '__main__':
    if iniciar_servidor():
        # Iniciar servidor Flask
        app.run(
            host='0.0.0.0',  # Accesible desde cualquier IP
            port=5000,        # Puerto estándar para APIs
            debug=True,       # Modo debug para desarrollo
            threaded=True     # Manejar múltiples solicitudes
        )