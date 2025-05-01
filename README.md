# Sistema de Supervisión Automática de Lavado de Manos

Este proyecto implementa un sistema completo para la supervisión automática de lavado de manos usando visión artificial y aprendizaje profundo. Utiliza el modelo YOLOv8 Pose para la detección de keypoints corporales y un modelo TSM-GRU (Temporal Shift Module + GRU) para clasificar las fases del lavado.

![Hand Washing Classification](https://www.cdc.gov/handwashing/images/wash-your-hands-steps-1280x720.jpg)

## Estructura del Proyecto

```
.
├── data/
│   ├── raw_videos/          # Videos originales del dataset
│   ├── keypoints_train/     # Keypoints extraídos para entrenamiento (.npy)
│   └── keypoints_val/       # Keypoints extraídos para validación (.npy)
├── docker/
│   └── Dockerfile           # Configuración para despliegue en Docker
├── model/
│   ├── temporal_shift.py    # Implementación del módulo Temporal Shift
│   └── tsm_gru.py           # Modelo TSM-GRU para clasificación de fases
├── scripts/
│   └── extract_keypoints.py # Script para extraer keypoints de videos
├── evaluate.py              # Script para evaluar el modelo entrenado
├── inference.py             # Script para inferencia en tiempo real
├── requirements.txt         # Dependencias del proyecto
└── train_tsm_gru.py         # Script para entrenar el modelo
```

## Instalación

1. **Instalar dependencias** (desde la raíz del proyecto):
   ```bash
   pip install -r lavadodemanosyoloposev11yentrenamiento/requirements.txt
   ```

2. **Dependencias específicas**:
   ```
   torch>=2.0.0
   torchvision>=0.15.0
   ultralytics>=8.0.0
   opencv-python>=4.7.0
   numpy>=1.24.0
   pandas>=2.0.0
   matplotlib>=3.7.0
   seaborn>=0.12.0
   scikit-learn>=1.2.0
   tqdm>=4.65.0
   paho-mqtt>=2.2.0
   ```

   > **Nota**: Para versiones específicas de PyTorch con soporte CUDA, instala PyTorch por separado siguiendo las instrucciones en pytorch.org.

## Implementación de Modelos

### 1. YOLOv8 Pose

Este proyecto utiliza el modelo preentrenado YOLOv8 Pose de Ultralytics para detectar keypoints corporales en cada frame de video.

- Modelo: `yolov8n-pose.pt` (versión nano para mayor velocidad)
- Keypoints detectados: 17 keypoints de cuerpo completo
- **Keypoints Seleccionados**: Para mejorar la relevancia, solo se utilizan 7 keypoints: nariz, hombros, codos y muñecas (índices 0, 5, 6, 7, 8, 9, 10).
- **Normalización**: Las coordenadas (x, y) de los keypoints seleccionados se normalizan restando el punto medio de los hombros para mayor robustez.
- **Dimensión de características**: 21 (7 keypoints × 3 valores [nx, ny, confianza])

### 2. Modelo TSM-GRU

El modelo de clasificación temporal combina:

- **Módulo de Desplazamiento Temporal (TSM)**: Permite la captura de información temporal al intercambiar parcialmente características entre frames adyacentes.
- **Unidad Recurrente GRU**: Procesa secuencias largas capturando dependencias temporales.

## Preparación de Datos

1. **Descargar el dataset** (desde la raíz del proyecto):
   ```bash
   # Crear directorio para los videos (si no existe)
   mkdir -p lavadodemanosyoloposev11yentrenamiento/data/raw_videos

   # Descargar el Hand Wash Dataset desde Kaggle o cualquier otra fuente
   # y colocar los videos en lavadodemanosyoloposev11yentrenamiento/data/raw_videos/HandWashDataset/
   ```

2. **Extracción de Keypoints** (desde la raíz del proyecto):
   ```bash
   python lavadodemanosyoloposev11yentrenamiento/scripts/extract_keypoints.py
   ```
   
   Este script:
   - Lee los videos del dataset
   - Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) para mejorar contraste
   - Extrae 17 keypoints por frame usando YOLOv8 Pose
   - **Selecciona y normaliza los 7 keypoints relevantes (nariz, hombros, codos, muñecas)**
   - Guarda los keypoints procesados (21 características) como archivos .npy en las carpetas train/val
   - Crea archivos labels.csv con las etiquetas de clase

   Parámetros opcionales (ajustar rutas si es necesario):
   ```
   # Example: Run from root, specifying paths relative to the script's expected location
   python lavadodemanosyoloposev11yentrenamiento/scripts/extract_keypoints.py --raw-dir lavadodemanosyoloposev11yentrenamiento/data/raw_videos/HandWashDataset --train-dir lavadodemanosyoloposev11yentrenamiento/data/keypoints_train --val-dir lavadodemanosyoloposev11yentrenamiento/data/keypoints_val --model-path lavadodemanosyoloposev11yentrenamiento/yolov8n-pose.pt
   ```

## Entrenamiento

(Desde la raíz del proyecto):
```bash
python lavadodemanosyoloposev11yentrenamiento/train_tsm_gru.py
```

El modelo se entrenará utilizando:
- Arquitectura TSM-GRU (Temporal Shift Module + GRU)
- Optimizador Adam (lr=1e-4, weight_decay=1e-5)
- CrossEntropyLoss
- 50 épocas por defecto
- Métrica F1 para monitorizar rendimiento
- El mejor modelo se guardará como `best_tsm_gru.pth`

Parámetros opcionales (ajustar rutas si es necesario):
```
# Example: Run from root, specifying paths relative to the script's expected location
python lavadodemanosyoloposev11yentrenamiento/train_tsm_gru.py --train-dir lavadodemanosyoloposev11yentrenamiento/data/keypoints_train --val-dir lavadodemanosyoloposev11yentrenamiento/data/keypoints_val --save-path lavadodemanosyoloposev11yentrenamiento/best_tsm_gru.pth
--seq-len           # Longitud de secuencia (default: 150 frames)
--batch-size        # Tamaño de batch (default: 16)
--epochs            # Número de épocas (default: 50)
--lr                # Learning rate (default: 1e-4)
--weight-decay      # Weight decay (default: 1e-5)
--use-class-weights # Usar pesos de clase
--use-lr-scheduler  # Usar scheduler de LR
--input-dim         # Dimensión de entrada (default: 21 para 7 keypoints seleccionados)
--hidden-dim        # Dimensión oculta GRU (default: 256)
--gru-layers        # Número de capas GRU (default: 2)
--tsm-segments      # Segmentos TSM (default: 8)
--tsm-shift-div     # Divisor TSM (default: 3)
--dropout           # Dropout (default: 0.5)
--bidirectional     # Usar GRU bidireccional
--use-attention     # Usar atención
```

## Evaluación

(Desde la raíz del proyecto):
```bash
python lavadodemanosyoloposev11yentrenamiento/evaluate.py --bidirectional --use-attention
```

Esto generará:
- Una matriz de confusión (`confusion_matrix.png`)
- Un reporte de clasificación con precisión, recall y F1-score para cada fase

**Rendimiento Obtenido (Ejemplo)**:
Con la configuración final (7 keypoints normalizados, TSM-GRU bidireccional con atención, y parámetros optimizados), se alcanzó una **precisión del 85%** en el conjunto de validación.

Parámetros opcionales (ajustar rutas si es necesario):
```
# Example: Run from root, specifying paths relative to the script's expected location
python lavadodemanosyoloposev11yentrenamiento/evaluate.py --model-path lavadodemanosyoloposev11yentrenamiento/best_tsm_gru.pth --val-dir lavadodemanosyoloposev11yentrenamiento/data/keypoints_val --cm-path lavadodemanosyoloposev11yentrenamiento/confusion_matrix_eval.png
# Fallback args (bidirectional, use-attention, hidden-dim, etc.)
```

## Inferencia en Tiempo Real

(Desde la raíz del proyecto):
```bash
python lavadodemanosyoloposev11yentrenamiento/inference.py --show-video
```

Funcionalidades:
- Captura video desde webcam
- Aplica CLAHE y extrae keypoints usando YOLOv8 Pose
- Clasifica fases en ventanas deslizantes de 5 segundos
- Emite alertas si la fase de enjabonado dura menos de 20 segundos
- Opcionalmente publica eventos vía MQTT

Parámetros opcionales (ajustar rutas si es necesario):
```
# Example: Run from root, specifying paths relative to the script's expected location
python lavadodemanosyoloposev11yentrenamiento/inference.py --model-path lavadodemanosyoloposev11yentrenamiento/best_tsm_gru.pth --yolo-model lavadodemanosyoloposev11yentrenamiento/YOLO11n-pose.pt --webcam-index      # Índice de la cámara (default: 0)
--mqtt-broker       # Dirección del broker MQTT (default: localhost)
--mqtt-port         # Puerto del broker MQTT (default: 1883)
--show-video        # Mostrar video con predicciones
```

## Etiquetas de Clases
- 0: Mojar
- 1: Enjabonado
- 2: Frotado
- 3: Aclarado

## Cambios Respecto al Modelo Original

Este proyecto utiliza YOLOv8 Pose para extraer keypoints corporales en lugar del modelo YOLOv11 mencionado originalmente, que puede no estar disponible públicamente. La implementación actual:

- Usa `yolov8n-pose.pt` de Ultralytics
- Procesa 7 keypoints seleccionados y normalizados (21 características)
- Es compatible con la biblioteca Ultralytics estándar
- Requiere PyTorch y CUDA para una ejecución óptima

## Despliegue con Docker

(Desde la raíz del proyecto):
```bash
# Construir la imagen
docker build -t handwash-monitor -f lavadodemanosyoloposev11yentrenamiento/docker/Dockerfile .

# Ejecutar el contenedor (ejemplo para Linux con webcam en /dev/video0)
# Ajusta --device según tu sistema operativo si es necesario
docker run --rm -it --device=/dev/video0:/dev/video0 handwash-monitor
```

## Créditos

Este proyecto utiliza el Hand Wash Dataset, diseñado para reconocer las diferentes fases del lavado de manos siguiendo las pautas de la OMS.