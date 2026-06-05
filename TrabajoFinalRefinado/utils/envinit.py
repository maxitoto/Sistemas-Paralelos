import os
import json
import sys
import shutil

def inicializar_entorno(config_path='config.json'):
    """
    Carga la configuración global y garantiza que la estructura 
    de directorios físicos exista antes de iniciar el orquestador.
    Cualquier error crítico aborta la ejecución de forma segura.
    """
    # 1. Verificar existencia del archivo
    if not os.path.exists(config_path):
        print(f"❌ [Error Crítico] No se encontró el archivo de configuración: '{config_path}'")
        sys.exit(1)
        
    # 2. Leer y parsear el JSON de forma segura
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ [Error Crítico] El archivo '{config_path}' tiene un formato JSON inválido.\n   Detalle: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ [Error Crítico] Fallo inesperado al leer '{config_path}'.\n   Detalle: {e}")
        sys.exit(1)

    # 3. Extraer el bloque de rutas (con validación de tipos)
    if not isinstance(config, dict):
        print("❌ [Error Crítico] La configuración no tiene una estructura válida (se esperaba un Diccionario).")
        sys.exit(1)

    rutas = config.get("paths", {})
    if not rutas:
        print("⚠️ [Advertencia] No se encontró la sección 'paths' en config.json. No se crearán carpetas automáticamente.")

    # 4. Validar y crear directorios
    for ruta in rutas.values():
        try:
            if not os.path.exists(ruta):
                os.makedirs(ruta)
                print(f"🔧 [Entorno] Directorio creado: {ruta}/")
        except PermissionError:
            print(f"❌ [Error Crítico] Permisos denegados por el Sistema Operativo para crear la carpeta: '{ruta}'")
            sys.exit(1)
        except Exception as e:
            print(f"❌ [Error Crítico] No se pudo crear el directorio '{ruta}'.\n   Detalle: {e}")
            sys.exit(1)

    return config



def limpiar_temporales(config):
    """
    Borra absolutamente todo el contenido de static/temp (archivos y subcarpetas)
    de forma segura, sin detener el programa si un archivo está bloqueado.
    """
    try:
        temp_dir = config.get("paths", {}).get("temp_dir")
        
        if not temp_dir or not os.path.exists(temp_dir):
            return # Si no existe, no hay nada que limpiar

        elementos_borrados = 0
        
        for elemento in os.listdir(temp_dir):
            ruta_elemento = os.path.join(temp_dir, elemento)
            try:
                # Si es un archivo suelto, lo borramos con os.remove
                if os.path.isfile(ruta_elemento) or os.path.islink(ruta_elemento):
                    os.remove(ruta_elemento)
                # Si es una subcarpeta (ej. 'frames' o 'audio'), la borramos entera con shutil
                elif os.path.isdir(ruta_elemento):
                    shutil.rmtree(ruta_elemento)
                    
                elementos_borrados += 1
                
            except PermissionError:
                print(f"⚠️ [Advertencia] El elemento '{elemento}' está en uso o protegido. No se pudo borrar.")
            except Exception as e:
                print(f"⚠️ [Advertencia] Fallo al borrar '{elemento}'. Detalle: {e}")
        
        if elementos_borrados > 0:
            print(f"🧹 [Entorno] Carpeta temporal vaciada por completo.")
            
    except Exception as e:
        print(f"⚠️ [Error Leve] Problema inesperado en la limpieza general de temporales.\n   Detalle: {e}")