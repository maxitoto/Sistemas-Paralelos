# se encarga de dar formato y emitir un correcto csv con las etadisticas
import csv
import os

def guardar_reporte(ruta_archivo, head, cuerpo):
    """
    Funcionalidad Aislada: Exportador a CSV.
    Recibe una lista de cabeceras (head) y una lista de listas (cuerpo) con los datos,
    y los escribe de forma segura en disco.
    """
    # 1. Asegurarnos de que la carpeta de destino (res/) exista antes de escribir
    directorio = os.path.dirname(ruta_archivo)
    if directorio and not os.path.exists(directorio):
        os.makedirs(directorio)
        
    print(f"📄 [Reporte] Generando archivo CSV...")
    
    try:
        # 2. Apertura segura:
        # encoding='utf-8' evita caracteres rotos en palabras como "Método" o "Cómputo"
        # newline='' es obligatorio para que no se inserten filas en blanco fantasma entre cada registro
        with open(ruta_archivo, mode='w', encoding='utf-8', newline='') as f:
            escritor = csv.writer(f)
            
            # 3. Escribir la cabecera (Head)
            if head:
                # Comprobamos si el head es un bloque de varias líneas (lista de listas) 
                # o una cabecera simple de una sola línea (lista de strings)
                if isinstance(head[0], list):
                    escritor.writerows(head)
                else:
                    escritor.writerow(head)
                    
            # 4. Escribir los datos (Cuerpo)
            if cuerpo:
                escritor.writerows(cuerpo)
                
        print(f"✅ [Reporte] Resultados guardados exitosamente en: {ruta_archivo}")
        return True
        
    except PermissionError:
        print(f"❌ [Error Crítico] Permiso denegado para escribir en '{ruta_archivo}'.")
        print("💡 Consejo: Asegúrate de no tener el archivo CSV abierto en Excel mientras corres el script.")
        return False
    except Exception as e:
        print(f"❌ [Error Crítico] Fallo inesperado al exportar el CSV. Detalle: {e}")
        return False