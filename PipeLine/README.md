```python?code_reference&code_event_index=3
# Contenido del README.md basado en el análisis de benchmark.py
readme_content = """# Image Processing Benchmark Orchestrator

Este proyecto es un **Orquestador de Benchmarking** diseñado para evaluar y comparar el rendimiento de diferentes estrategias de procesamiento de imágenes (Filtro Sobel). El sistema permite medir de forma precisa el impacto de optimizaciones como la vectorización con **NumPy** y la compilación JIT/paralelismo con **Numba** frente a ejecuciones secuenciales.

## 🚀 Características principales

- **Ejecución Modular:** Carga dinámicamente "pipelines" organizados en carpetas independientes.
- **Pipeline de 3 Fases:**
  1. **Fase 1:** Carga y preprocesamiento (Conversión a Grises).
  2. **Fase 2:** Procesamiento core (Filtro Sobel).
  3. **Fase 3:** Post-procesamiento y validación (Cálculo de blancos y guardado).
- **Estadísticas Robustas:** Soporta múltiples ejecuciones (`--runs`) para promediar tiempos y minimizar el ruido del sistema.
- **Cálculo de Speed-up:** Compara automáticamente el rendimiento contra una línea base (baseline) secuencial.
- **Reportes Automáticos:** Genera archivos `reporte_estadistico.csv` con métricas de tiempo, hardware y eficiencia.

## 🛠️ Requisitos

- Python 3.x
- **Pillow** (PIL) para el manejo de imágenes.
- **NumPy** y **Numba** (opcionales, dependiendo de las estrategias a testear).

## 💻 Guía de Uso

El orquestador se controla totalmente desde la línea de comandos.

### Comando básico
```bash
python3 benchmark.py --input <ruta_imagen> --pipeline <nombre_carpeta> --runs 5
```

### Argumentos disponibles
| Parámetro | Descripción | Default |
| :--- | :--- | :--- |
| `--input` | Ruta de la imagen a procesar (ej: `img/paisaje.png`). | Requerido |
| `--pipeline` | Lista de estrategias separadas por coma (ej: `secuencial,numpy`). | Requerido |
| `--runs` | Cantidad de iteraciones para promediar el tiempo. | 1 |
| `--save_baseline`| Si se usa con el pipeline `secuencial`, guarda el tiempo como referencia. | False |
| `--contexto` | Nombre descriptivo para el reporte (ej: "Test en MacBook Pro"). | "Prueba" |

### Ejemplo práctico
```bash
# 1. Establecer el tiempo base con el método secuencial
python3 benchmark.py --input foto.png --pipeline secuencial --runs 5 --save_baseline

# 2. Medir la mejora de rendimiento con Numba
python3 benchmark.py --input foto.png --pipeline numba_paralelo --runs 5
```

Aquí tienes la sección terminada, manteniendo la estructura técnica y profesional que veníamos trabajando:

## 📊 Salida de Datos

Los resultados numéricos y estadísticos se centralizan en la carpeta `resultados/`:
* **`reporte_estadistico.csv`**: Un log detallado que registra cada experimento, incluyendo tiempos por fase, tiempo total, hardware detectado, métricas de **Speed-up** y el **% de píxeles blancos** para garantizar la consistencia entre métodos.
* **`tiempos_base.json`**: Almacena de forma persistente los récords de tiempo del pipeline secuencial, sirviendo como punto de referencia para los cálculos de aceleración.

Por otro lado, los artefactos visuales se organizan en subcarpetas jerárquicas según el pipeline y la imagen procesada (ej. `numba_cpu/mi_imagen/`):
* **`imagen_gris.png`**: Resultado de la Fase 1 (conversión a escala de grises y preprocesamiento).
* **`gx_verticales.png`**: Representación de los bordes detectados en el eje vertical mediante la máscara $G_x$.
* **`gy_horizontales.png`**: Representación de los bordes detectados en el eje horizontal mediante la máscara $G_y$.
* **`sobel_final.png`**: Imagen final resultante de la magnitud del gradiente ($\sqrt{G_x^2 + G_y^2}$), donde se aplica el umbral de detección de bordes.

---

> **Nota de validación:** El sistema está diseñado para que todas las imágenes de salida sean visualmente idénticas entre estrategias, permitiendo que la diferencia solo resida en el tiempo de cómputo y la eficiencia del uso de recursos. (solo se crean las imagenes la primera vez)