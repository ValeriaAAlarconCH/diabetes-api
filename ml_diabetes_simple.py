from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import traceback
import sys

app = Flask(__name__)
CORS(app)

# Variables globales
modelo = None
config = None

def cargar_modelo():
    """Carga el modelo y configuración de forma segura"""
    global modelo, config
    
    try:
        print("🚀 Cargando modelo de diabetes...")
        
        # 1. Cargar configuración primero
        with open('model_config.pkl', 'rb') as f:
            config = pickle.load(f)
        
        print(f"✅ Configuración cargada:")
        print(f"   - Features: {len(config.get('feature_names', []))}")
        print(f"   - Targets: {len(config.get('target_names', []))}")
        
        # 2. Cargar el modelo corregido
        model_file = 'modelo_diabetes_corregido.pkl'
        with open(model_file, 'rb') as f:
            modelo = pickle.load(f)
        
        print(f"✅ Modelo cargado desde: {model_file}")
        print(f"📊 Tipo de modelo: {type(modelo)}")
        
        # 3. Verificar que no tenga atributos problemáticos
        if hasattr(modelo, 'use_label_encoder'):
            print("⚠️  Eliminando atributo problemático: use_label_encoder")
            delattr(modelo, 'use_label_encoder')
        
        return True
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {str(e)}")
        print("💡 Verifica que los archivos existan:")
        print("   - model_config.pkl")
        print("   - modelo_diabetes_corregido.pkl")
        traceback.print_exc()
        return False

def preparar_datos(datos_json):
    """Prepara los datos para el modelo"""
    
    # Obtener nombres de características del config
    feature_names = config.get('feature_names', [])
    categorical_mapping = config.get('categorical_mapping', {})
    
    # Crear array de características
    features = np.zeros(len(feature_names))
    
    # Mapeo de nombres en español a inglés (si es necesario)
    spanish_to_english = {
        'edad': 'edad',
        'niveles_glucosa': 'niveles_glucosa',
        'niveles_insulina': 'niveles_insulina',
        'indice_masa_corporal': 'indice_masa_corporal',
        'autoanticuerpos': 'autoanticuerpos',
        'antecedentes_familiares': 'antecedentes_familiares',
        'marcadores_geneticos': 'marcadores_geneticos',
        'presion_arterial': 'presion_arterial',
        'niveles_colesterol': 'niveles_colesterol',
        'circunferencia_cintura': 'circunferencia_cintura',
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
        'factores_socioeconomicos': 'factores_socioeconomicos',
        'niveles_enzimas_digestivas': 'niveles_enzimas_digestivas',
        'peso_nacimiento': 'peso_nacimiento',
        'salud_pancreatica': 'salud_pancreatica',
        'funcion_pulmonar': 'funcion_pulmonar',
        'evaluaciones_neurologicas': 'evaluaciones_neurologicas',
        'aumento_peso_embarazo': 'aumento_peso_embarazo'
    }
    
    # Para cada característica esperada
    for i, feature_name in enumerate(feature_names):
        valor = None
        
        # 1. Buscar coincidencia exacta
        if feature_name in datos_json:
            valor = datos_json[feature_name]
        
        # 2. Buscar en mapeo español-inglés
        else:
            for spanish_key, english_key in spanish_to_english.items():
                if english_key == feature_name and spanish_key in datos_json:
                    valor = datos_json[spanish_key]
                    break
        
        # 3. Convertir valor
        if valor is not None:
            try:
                # Si es variable categórica y tenemos mapeo
                if feature_name in categorical_mapping:
                    if valor in categorical_mapping[feature_name]:
                        features[i] = categorical_mapping[feature_name][valor]
                    else:
                        features[i] = 0.0
                else:
                    # Variable numérica
                    features[i] = float(valor)
            except:
                features[i] = 0.0
    
    return features.reshape(1, -1)

@app.route('/health', methods=['GET'])
def health_check():
    """Verificar estado del servicio"""
    if modelo is not None and config is not None:
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'model_type': str(type(modelo)).split('.')[-1].replace("'>", ""),
            'num_features': len(config.get('feature_names', [])),
            'num_classes': len(config.get('target_names', [])),
            'classes': config.get('target_names', []),
            'features_sample': config.get('feature_names', [])[:5]
        })
    else:
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False,
            'message': 'Modelo no cargado'
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Realizar predicción"""
    try:
        if modelo is None or config is None:
            return jsonify({
                'error': 'Modelo no cargado',
                'success': False
            }), 500
        
        datos = request.json
        
        if not datos:
            return jsonify({
                'error': 'No se recibieron datos',
                'success': False
            }), 400
        
        print(f"📥 Datos recibidos: {list(datos.keys())}")
        
        # Preparar datos
        datos_preparados = preparar_datos(datos)
        
        # Hacer predicción
        try:
            prediccion_numerica = modelo.predict(datos_preparados)[0]
            prediccion_proba = modelo.predict_proba(datos_preparados)[0]
        except Exception as e:
            print(f"⚠️  Error en predicción: {e}")
            # Fallback a predicción simple basada en reglas
            return prediccion_fallback(datos)
        
        # Obtener nombres de clases
        target_names = config.get('target_names', [])
        
        if prediccion_numerica < len(target_names):
            clase_predicha = target_names[prediccion_numerica]
            probabilidad = float(prediccion_proba[prediccion_numerica])
        else:
            clase_predicha = "Unknown"
            probabilidad = 0.0
        
        # Todas las probabilidades
        probabilidades = {}
        for i, clase in enumerate(target_names):
            if i < len(prediccion_proba):
                probabilidades[clase] = float(prediccion_proba[i])
        
        # Importancia de características (si está disponible)
        importancia = {}
        if hasattr(modelo, 'feature_importances_'):
            feature_names = config.get('feature_names', [])
            for i, feature in enumerate(feature_names):
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
        print(f"❌ Error en predicción: {str(e)}")
        traceback.print_exc()
        
        return jsonify({
            'error': str(e),
            'success': False,
            'message': 'Error en la predicción'
        }), 500

def prediccion_fallback(datos):
    """Predicción de respaldo cuando el modelo falla"""
    print("🎭 Usando predicción de respaldo...")
    
    # Extraer valores importantes
    edad = datos.get('edad', datos.get('age', 45))
    glucosa = datos.get('niveles_glucosa', datos.get('glucosa', 100))
    
    # Reglas simples
    if glucosa > 200:
        clase = 'Type 1 Diabetes'
        prob = 0.85
    elif glucosa > 126:
        clase = 'Type 2 Diabetes'
        prob = 0.75
    elif glucosa >= 100:
        clase = 'Prediabetic'
        prob = 0.65
    else:
        clase = 'Gestational Diabetes'
        prob = 0.50
    
    # Crear probabilidades
    target_names = config.get('target_names', [
        'Type 1 Diabetes', 'Type 2 Diabetes', 'Prediabetic',
        'Gestational Diabetes'
    ])
    
    probabilidades = {}
    for target in target_names:
        if target == clase:
            probabilidades[target] = prob
        else:
            probabilidades[target] = (1 - prob) / (len(target_names) - 1)
    
    return jsonify({
        'predictedClass': clase,
        'probability': prob,
        'probabilities': probabilidades,
        'success': True,
        'message': 'Predicción usando respaldo (modelo principal falló)'
    })

@app.route('/test', methods=['POST', 'GET'])
def test_prediction():
    """Endpoint de prueba"""
    
    # Datos de ejemplo
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
        'prueba_orina': 'Normal'
    }
    
    if request.method == 'POST':
        # Usar el endpoint de predicción normal
        request._json = datos_ejemplo
        return predict()
    else:
        # Mostrar información
        return jsonify({
            'message': 'Use POST /test para probar o POST /predict con tus datos',
            'example_data': datos_ejemplo
        })

@app.route('/features', methods=['GET'])
def get_features():
    """Obtener información de características"""
    if config is None:
        return jsonify({'error': 'Configuración no cargada'}), 500
    
    return jsonify({
        'features': config.get('feature_names', []),
        'target_classes': config.get('target_names', []),
        'categorical_features': config.get('categorical_mapping', {}),
        'num_features': len(config.get('feature_names', [])),
        'num_classes': len(config.get('target_names', []))
    })

@app.route('/config', methods=['GET'])
def get_config():
    """Obtener configuración completa"""
    if config is None:
        return jsonify({'error': 'Configuración no cargada'}), 500
    
    return jsonify(config)

if __name__ == '__main__':
    print("=" * 60)
    print("   🎯 API de Predicción de Diabetes - VERSIÓN FINAL")
    print("=" * 60)
    
    if cargar_modelo():
        print("\n✅ API INICIADA CORRECTAMENTE")
        print("📡 Endpoints disponibles:")
        print("   GET  /health     - Estado del servicio")
        print("   GET  /features   - Características del modelo")
        print("   GET  /config     - Configuración completa")
        print("   POST /predict    - Realizar predicción")
        print("   GET  /test       - Información de prueba")
        print("   POST /test       - Prueba con datos de ejemplo")
        print("\n🌐 URL: http://localhost:5000")
        print("\n📊 Información del modelo:")
        print(f"   - Features: {len(config.get('feature_names', []))}")
        print(f"   - Clases: {len(config.get('target_names', []))}")
        print(f"   - Tipo: {type(modelo)}")
        
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("\n❌ NO SE PUDO INICIAR LA API")
        print("\n💡 SOLUCIONES:")
        print("   1. Verifica que existan los archivos:")
        print("      - model_config.pkl")
        print("      - modelo_diabetes_corregido.pkl")
        print("   2. Ejecuta: python -c \"import pickle; print('OK')\"")
        print("   3. Revisa los permisos de los archivos")