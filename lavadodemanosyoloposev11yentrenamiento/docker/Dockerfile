# Usa una imagen base de Python
FROM python:3.10-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de requerimientos primero para aprovechar el caché de Docker
COPY requirements.txt requirements.txt

# Instala las dependencias
# Es recomendable instalar PyTorch primero, eligiendo la versión CPU o GPU adecuada
# Aquí usamos la versión CPU por defecto para compatibilidad general
# Si necesitas GPU, busca la imagen base de NVIDIA (e.g., nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04) 
# y ajusta la instalación de torch.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# Instala el resto de dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código del proyecto al directorio de trabajo
# Copia directorios específicos necesarios para la inferencia
COPY model /app/model
COPY scripts /app/scripts # Si inference.py usa algo de scripts (ej. funciones auxiliares)
COPY inference.py /app/inference.py
# Copia el modelo entrenado (si ya existe)
COPY best_tsm_gru.pth /app/best_tsm_gru.pth
# Copia el modelo YOLO (si lo tienes localmente y no quieres descargarlo en el contenedor)
COPY yolo11n-pose.pt /app/yolo11n-pose.pt

# Expón el puerto MQTT si el contenedor necesita recibir conexiones (no es el caso aquí)
# EXPOSE 1883

# Define el punto de entrada para ejecutar el script de inferencia
# Pasa la ruta del modelo como argumento
ENTRYPOINT ["python", "inference.py", "--model-path", "/app/best_tsm_gru.pth", "--yolo-model", "/app/yolo11n-pose.pt"]

# Opcional: Puedes añadir argumentos adicionales que inference.py acepte
# CMD ["--show-video"] # Ejemplo: si quieres pasar argumentos por defecto
