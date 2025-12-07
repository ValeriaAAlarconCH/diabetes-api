# fix_model.py
import pickle
import xgboost as xgb

print("🔧 Corrigiendo modelo XGBoost...")

try:
    # 1. Cargar el modelo
    with open('modelo_diabetes.pkl', 'rb') as f:
        modelo = pickle.load(f)
    
    print(f"✅ Modelo cargado. Tipo: {type(modelo)}")
    
    # 2. Si es XGBClassifier, corregir atributos problemáticos
    if hasattr(modelo, '__class__') and 'XGBClassifier' in str(modelo.__class__):
        print("📝 Identificado como XGBClassifier, corrigiendo...")
        
        # Eliminar atributo problemático si existe
        if hasattr(modelo, 'use_label_encoder'):
            print("🗑️  Eliminando use_label_encoder...")
            delattr(modelo, 'use_label_encoder')
        
        # Asegurarse de que los parámetros sean correctos
        if hasattr(modelo, 'get_params'):
            # Obtener parámetros sin usar get_params (para evitar el error)
            params = modelo.__dict__.copy()
            
            # Si hay otros atributos problemáticos, manejarlos
            for key in list(params.keys()):
                if '_le' in key or 'label_encoder' in key.lower():
                    print(f"🗑️  Eliminando atributo problemático: {key}")
                    delattr(modelo, key)
    
    # 3. Guardar el modelo corregido
    with open('modelo_diabetes_corregido.pkl', 'wb') as f:
        pickle.dump(modelo, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("✅ Modelo corregido guardado como 'modelo_diabetes_corregido.pkl'")
    
    # 4. Verificar que se puede cargar
    print("🧪 Verificando modelo corregido...")
    with open('modelo_diabetes_corregido.pkl', 'rb') as f:
        modelo_corregido = pickle.load(f)
    
    print(f"✅ Modelo corregido cargado exitosamente")
    print(f"📊 Tipo: {type(modelo_corregido)}")
    
    # Intentar obtener parámetros
    try:
        print("📝 Intentando obtener parámetros...")
        if hasattr(modelo_corregido, 'get_params'):
            params = modelo_corregido.get_params()
            print(f"✅ Parámetros obtenidos: {len(params)} parámetros")
    except Exception as e:
        print(f"⚠️  No se pueden obtener parámetros, pero el modelo está cargado: {e}")
    
except Exception as e:
    print(f"❌ Error corrigiendo modelo: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("INSTRUCCIONES:")
print("1. Copia el archivo 'modelo_diabetes_corregido.pkl' a 'modelo_diabetes.pkl'")
print("2. O actualiza tu código para usar 'modelo_diabetes_corregido.pkl'")
print("="*50)