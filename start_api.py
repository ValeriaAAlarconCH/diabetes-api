import subprocess
import sys
import os

def verificar_archivos():
    """Verifica que todos los archivos necesarios existan"""
    archivos_necesarios = ['ml_diabetes_simple.py', 'requirements_final.txt']
    archivos_modelo = ['model_config.pkl', 'modelo_diabetes_corregido.pkl']
    
    print("🔍 Verificando archivos necesarios...")
    
    for archivo in archivos_necesarios:
        if not os.path.exists(archivo):
            print(f"❌ Faltante: {archivo}")
            return False
    
    # Verificar al menos un archivo de modelo
    tiene_modelo = any(os.path.exists(f) for f in archivos_modelo)
    if not tiene_modelo:
        print("❌ No se encontró ningún archivo de modelo (.pkl)")
        return False
    
    print("✅ Todos los archivos necesarios encontrados")
    return True

def instalar_dependencias():
    """Instala las dependencias necesarias"""
    print("📦 Instalando/verificando dependencias...")
    
    try:
        # Verificar si pip está disponible
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        
        # Instalar dependencias
        resultado = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements_final.txt"],
            capture_output=True,
            text=True
        )
        
        if resultado.returncode == 0:
            print("✅ Dependencias instaladas correctamente")
            return True
        else:
            print(f"⚠️  Problemas instalando dependencias: {resultado.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def main():
    """Función principal"""
    print("=" * 50)
    print("   INICIADOR DE API DE DIABETES")
    print("=" * 50)
    
    # Verificar archivos
    if not verificar_archivos():
        print("\n💡 Soluciones:")
        print("   1. Asegúrate de estar en la carpeta correcta")
        print("   2. Ejecuta fix_model.py para corregir el modelo")
        print("   3. Verifica que los archivos .pkl existan")
        return
    
    # Instalar dependencias si es necesario
    if not instalar_dependencias():
        print("⚠️  Continuando sin verificar dependencias...")
    
    # Iniciar la API
    print("\n🚀 Iniciando API de Diabetes...")
    print("🌐 La API estará disponible en: http://localhost:5000")
    print("🛑 Presiona CTRL+C para detener\n")
    
    try:
        # Importar y ejecutar la API
        from ml_diabetes_simple import iniciar_servidor
        if iniciar_servidor():
            from ml_diabetes_simple import app
            app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n👋 API detenida por el usuario")
    except Exception as e:
        print(f"\n❌ Error al iniciar la API: {e}")

if __name__ == "__main__":
    main()