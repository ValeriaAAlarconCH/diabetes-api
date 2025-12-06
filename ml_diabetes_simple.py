from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Variables globales
modelo = None
config = None

def cargar_modelo():
    global modelo, config
    
    try:
        print("🚀 Cargando modelo de diabetes...")
        
        # 1. Cargar el modelo
        with open('modelo_diabetes.pkl', 'rb') as f:
            modelo = pickle.load(f)
        print(f"✅ Modelo cargado. Tipo: {type(modelo)}")
        
        # 2. Cargar configuración
        with open('model_config.pkl', 'rb') as f:
            config = pickle.load(f)
        print(f"✅ Configuración cargada. {len(config['feature_names'])} features")
        
        return True
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {str(e)}")
        return False

def preparar_datos_simples(datos_json):
    """Prepara datos de forma simple sin encoders complejos"""
    
    # Crear un array vacío con el tamaño correcto
    num_features = len(config['feature_names'])
    datos_array = np.zeros(num_features)
    
    # Mapeo de nombres de columnas (por si hay diferencias)
    feature_map = {
        # Mapea nombres en español a nombres en inglés/features reales
        'marcadores_geneticos': 'marcadores_geneticos',
        'autoanticuerpos': 'autoanticuerpos',
        'antecedentes_familiares': 'antecedentes_familiares',
        'factores_ambientales': 'factores_ambientales',
        'etnicidad': 'etnicidad',
        'habitos_alimenticios': 'habitos_alimenticios',
        'prueba_tolerancia_glucosa': 'prueba_tolerancia_glucosa',
        'pruebas_funcion_hepatica': 'pruebas_funcion_hepatica',
        'diagnostico_fibrosis_quistica': 'diagnostico_fibrosis_quistica',
        'uso_esteroides': 'uso_esteroides',
        'pruebas_geneticas': 'pruebas_geneticas',
        'historial_embarazos': 'historial_embarazos',
        'diabetes_gestacional_previa': 'diabetes_gestacional_previa',
        'historial_pcos': 'historial_pcos',
        'estado_tabaquismo': 'estado_tabaquismo',
        'sintomas_inicio_temprano': 'sintomas_inicio_temprano',
        'factores_socioeconomicos': 'factores_socioeconomicos',
        'consumo_alcohol': 'consumo_alcohol',
        'actividad_fisica': 'actividad_fisica',
        'prueba_orina': 'prueba_orina',
        'niveles_insulina': 'niveles_insulina',
        'edad': 'edad',
        'indice_masa_corporal': 'indice_masa_corporal',
        'presion_arterial': 'presion_arterial',
        'niveles_colesterol': 'niveles_colesterol',
        'circunferencia_cintura': 'circunferencia_cintura',
        'niveles_glucosa': 'niveles_glucosa',
        'aumento_peso_embarazo': 'aumento_peso_embarazo',
        'salud_pancreatica': 'salud_pancreatica',
        'funcion_pulmonar': 'funcion_pulmonar',
        'evaluaciones_neurologicas': 'evaluaciones_neurologicas',
        'niveles_enzimas_digestivas': 'niveles_enzimas_digestivas',
        'peso_nacimiento': 'peso_nacimiento'
    }
    
    # Llenar el array con los datos recibidos
    for i, feature_name in enumerate(config['feature_names']):
        # Buscar el valor en los datos recibidos
        valor = None
        
        # Primero intentar con el nombre exacto
        if feature_name in datos_json:
            valor = datos_json[feature_name]
        else:
            # Buscar en el mapeo
            for key_es, key_en in feature_map.items():
                if key_en == feature_name and key_es in datos_json:
                    valor = datos_json[key_es]
                    break
        
        if valor is not None:
            # Convertir categórico a numérico si es necesario
            if isinstance(valor, str) and 'categorical_mapping' in config:
                for cat_col, mapping in config['categorical_mapping'].items():
                    if cat_col in feature_name.lower() or feature_name.lower() in cat_col:
                        if valor in mapping:
                            datos_array[i] = mapping[valor]
                        else:
                            # Valor por defecto
                            datos_array[i] = 0
                        break
                else:
                    # Si no es categórico, intentar convertir a número
                    try:
                        datos_array[i] = float(valor)
                    except:
                        datos_array[i] = 0.0
            else:
                try:
                    datos_array[i] = float(valor)
                except:
                    datos_array[i] = 0.0
    
    return datos_array.reshape(1, -1)

@app.route('/health', methods=['GET'])
def health_check():
    if modelo is not None:
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'model_type': str(type(modelo)),
            'num_features': len(config['feature_names']),
            'classes': config['target_names']
        })
    else:
        return jsonify({'status': 'unhealthy', 'model_loaded': False}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if modelo is None:
            return jsonify({'error': 'Modelo no cargado'}), 500
        
        # Obtener datos del request
        datos = request.json
        
        # Preparar datos
        datos_preparados = preparar_datos_simples(datos)
        
        # Hacer predicción
        prediccion_numerica = modelo.predict(datos_preparados)[0]
        prediccion_proba = modelo.predict_proba(datos_preparados)[0]
        
        # Obtener nombre de la clase
        clase_predicha = config['target_names'][prediccion_numerica]
        probabilidad = float(prediccion_proba[prediccion_numerica])
        
        # Todas las probabilidades
        probabilidades = {}
        for i, clase in enumerate(config['target_names']):
            probabilidades[clase] = float(prediccion_proba[i])
        
        # Importancia de características (si está disponible)
        importancia = {}
        if hasattr(modelo, 'feature_importances_'):
            for i, feature in enumerate(config['feature_names']):
                importancia[feature] = float(modelo.feature_importances_[i])
        
        # Respuesta
        respuesta = {
            'predictedClass': clase_predicha,
            'probability': probabilidad,
            'probabilities': probabilidades,
            'featureImportance': importancia,
            'success': True,
            'message': 'Predicción exitosa'
        }
        
        return jsonify(respuesta)
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error en predicción: {error_msg}")
        return jsonify({
            'error': error_msg,
            'success': False,
            'message': 'Error en la predicción'
        }), 500

@app.route('/test', methods=['POST'])
def test_prediction():
    """Endpoint de prueba con datos de ejemplo"""
    datos_ejemplo = {
        'edad': 45,
        'niveles_glucosa': 180,
        'niveles_insulina': 35,
        'indice_masa_corporal': 28.5,
        'autoanticuerpos': 'Negative',
        'antecedentes_familiares': 'Yes',
        'marcadores_geneticos': 'Positive',
        'factores_ambientales': 'Present',
        'etnicidad': 'High Risk',
        'habitos_alimenticios': 'Unhealthy',
        'prueba_tolerancia_glucosa': 'Abnormal',
        'presion_arterial': 130,
        'niveles_colesterol': 220,
        'circunferencia_cintura': 95,
        'actividad_fisica': 'Low'
    }
    
    # Usar el endpoint predict normal
    return predict_test(datos_ejemplo)

def predict_test(datos):
    """Función auxiliar para pruebas"""
    if modelo is None:
        return {'error': 'Modelo no cargado'}
    
    try:
        datos_preparados = preparar_datos_simples(datos)
        prediccion_numerica = modelo.predict(datos_preparados)[0]
        prediccion_proba = modelo.predict_proba(datos_preparados)[0]
        
        return {
            'clase': config['target_names'][prediccion_numerica],
            'probabilidad': float(prediccion_proba[prediccion_numerica]),
            'todas_probabilidades': {
                config['target_names'][i]: float(prediccion_proba[i]) 
                for i in range(len(config['target_names']))
            }
        }
    except Exception as e:
        return {'error': str(e)}

if __name__ == '__main__':
    print("=" * 50)
    print("   API de Predicción de Diabetes - Versión Simple")
    print("=" * 50)
    
    if cargar_modelo():
        print("\n✅ API lista para recibir solicitudes")
        print("📡 Endpoints disponibles:")
        print("   GET  /health  - Verificar estado")
        print("   POST /predict - Realizar predicción")
        print("   POST /test    - Prueba con datos de ejemplo")
        print("\n🌐 Servidor iniciado en: http://localhost:5000")
        print("📝 Para probar: curl -X POST http://localhost:5000/predict -H 'Content-Type: application/json' -d '{\"edad\": 45, \"niveles_glucosa\": 180}'")
        
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ No se pudo cargar el modelo. Verifica los archivos .pkl")
        print("📁 Asegúrate de tener en la misma carpeta:")
        print("   - modelo_diabetes.pkl")
        print("   - model_config.pkl")