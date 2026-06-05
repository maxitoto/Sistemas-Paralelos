# 🛠️ Requisitos del Sistema y Hardware (Dependencias Externas)

Para que el motor de procesamiento HPC (High Performance Computing) funcione correctamente, la arquitectura requiere ciertas herramientas a nivel del sistema operativo que **no** pueden instalarse mediante `pip` o el archivo `requirements.txt`.

Las siguientes instrucciones están orientadas a entornos **macOS**.

## 1. Gestor de Paquetes: Homebrew
Homebrew es el gestor de paquetes estándar para macOS. Te permitirá instalar las herramientas de consola necesarias con un solo comando.

* **¿Cómo verificar si ya lo tienes?** Abre tu terminal y ejecuta: `brew --version`
* **Si no lo tienes instalado**, ejecuta este comando en tu terminal:
  ```bash
  /bin/bash -c "$(curl -fsSL [https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh](https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh))"

```

## 2. Motor Multimedia: FFmpeg (Obligatorio)

FFmpeg es el framework multimedia líder en la industria. El orquestador de este proyecto se comunica con él en segundo plano (vía `subprocess`) para manipular el audio sin perder calidad.

* **¿Por qué lo necesitamos?** OpenCV (la librería de Python) es excelente manipulando matrices de imágenes, pero destruye las pistas de audio nativas al guardar un video. FFmpeg se encarga de extraer el audio original en la Fase de Pre-procesamiento y de inyectarlo en el video mudo resultante en la Fase de Ensamblado.
* **Instalación:**
```bash
brew install ffmpeg

```


* **Verificación:** Para asegurarte de que se instaló correctamente, ejecuta `ffmpeg -version`. Deberías ver un texto largo con la configuración de la herramienta.

## 3. Monitoreo de Recursos: htop (Opcional pero Recomendado)

Para la defensa del Trabajo Práctico y la redacción del informe de métricas, el profesor ha solicitado observar el comportamiento del sistema (uso de todos los núcleos lógicos) en tiempo real.

Aunque macOS incluye el comando nativo `top`, recomendamos instalar `htop`, que ofrece una interfaz visual mucho más amigable, indicando el uso de CPU hilo por hilo y el consumo de RAM con barras de colores.

* **Instalación:**
```bash
brew install htop

```


* **Uso durante el Benchmark:** 1. Abre una segunda ventana de tu terminal.
2. Escribe `htop` y presiona Enter.
3. Ejecuta el benchmark de Python en tu primera ventana y observa cómo los núcleos de tu procesador llegan al 100% de uso.
4. Presiona `q` para salir del monitor.

---

💡 **Nota sobre Aceleración por Hardware en Mac:** Este proyecto está preparado para ejecutarse en aceleradores gráficos. Si utilizas una Mac con procesadores Apple Silicon (M1/M2/M3), el orquestador y PyTorch utilizarán el backend **MPS (Metal Performance Shaders)** en lugar de CUDA, aprovechando la GPU integrada del sistema para el procesamiento tensorial.