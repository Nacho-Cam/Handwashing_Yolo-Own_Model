import cv2
import numpy as np
import os
import pandas as pd
from ultralytics import YOLO
import torch
from tqdm import tqdm
import argparse
import random

# --- Configuración ---
# Rutas relativas al script actual
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) # Directorio raíz del proyecto (lavadodemanosyoloposev11yentrenamiento)
DATA_DIR = os.path.join(BASE_DIR, 'data')
# Paths relative to BASE_DIR
RAW_VIDEOS_DIR = os.path.join(DATA_DIR, 'raw_videos', 'HandWashDataset')
KEYPOINTS_TRAIN_DIR = os.path.join(DATA_DIR, 'keypoints_train')
KEYPOINTS_VAL_DIR = os.path.join(DATA_DIR, 'keypoints_val')
LABELS_FILE_TRAIN = os.path.join(KEYPOINTS_TRAIN_DIR, 'labels.csv')
LABELS_FILE_VAL = os.path.join(KEYPOINTS_VAL_DIR, 'labels.csv')
# Model path relative to BASE_DIR
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, 'yolov8n-pose.pt') # Correct path to model in parent dir
DEFAULT_SPLIT_RATIO = 0.8 # 80% para entrenamiento, 20% para validación

# Número de keypoints que produce el modelo YOLOv8 pose (17 para cuerpo completo) 
# y dimensión de características por frame (17 kpts * 3 valores = 51)
YOLOV8_POSE_KEYPOINTS = 17
FEATURE_DIM = YOLOV8_POSE_KEYPOINTS * 3  # x, y, confidence para cada keypoint

# Mapeo de las fases para el Hand Wash Dataset de Kaggle
# 0=mojar, 1=enjabonado, 2=frotado, 3=aclarado
def get_label_from_filename(filepath):
    """
    Extrae la etiqueta numérica (0-3) de la ruta del archivo.
    Adaptada específicamente para el Hand Wash Dataset de Kaggle,
    considerando los nombres de archivo/carpeta observados.
    """
    # Extraer el nombre del archivo o el nombre del directorio padre
    filename_or_dir = os.path.basename(os.path.dirname(filepath)).lower() # Check parent dir first
    if not filename_or_dir.startswith('step'): # If parent dir isn't step_X, check filename
        filename_or_dir = os.path.basename(filepath).lower()

    # Mapeo basado en el prefijo "step_"
    if 'step_1' in filename_or_dir:
        return 0  # Mojar
    elif 'step_2' in filename_or_dir: # Catches step_2_left, step_2_right, etc.
        return 1  # Enjabonado
    elif any(f'step_{i}' in filename_or_dir for i in range(3, 7)): # Steps 3, 4, 5, 6
        return 2  # Frotado
    elif 'step_7' in filename_or_dir:
        return 3  # Aclarado

    print(f"Advertencia: No se pudo determinar la etiqueta para {filepath} (nombre/dir: {filename_or_dir}). Asignando -1.")
    print("Examina el archivo/carpeta y adapta la función 'get_label_from_filename' según corresponda.")
    return -1  # Etiqueta inválida

def extract_keypoints_from_video(video_path, model, clahe):
    """
    Extrae keypoints de un vídeo frame a frame usando YOLO-Pose.
    Aplica CLAHE a cada frame antes de la detección.
    Retorna una secuencia de keypoints (T, 51) o None si falla.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error al abrir el vídeo: {video_path}")
        return None

    keypoints_sequence = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Usar tqdm si quieres una barra de progreso por vídeo
    # pbar_frames = tqdm(total=frame_count, desc=f"Procesando {os.path.basename(video_path)}", leave=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir a escala de grises para CLAHE
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Aplicar CLAHE
        clahe_frame_gray = clahe.apply(gray)
        # Convertir de nuevo a BGR para YOLO
        clahe_frame_bgr = cv2.cvtColor(clahe_frame_gray, cv2.COLOR_GRAY2BGR)

        # Inferencia con YOLO-Pose
        results = model(clahe_frame_bgr, verbose=False, device=model.device)

        # Inicializar con ceros por si no hay detección
        frame_kpts = np.zeros(FEATURE_DIM)

        if results and len(results) > 0 and results[0].keypoints is not None and results[0].keypoints.shape[1] > 0:
            # Obtener los datos de keypoints
            # results[0].keypoints.data tiene forma [num_detections, num_keypoints, 3]
            # donde 3 representa [x, y, confidence]
            kpts_data = results[0].keypoints.data
            
            # Seleccionar la persona principal (la primera detección)
            if kpts_data.shape[0] > 0:
                kpts = kpts_data[0].cpu().numpy()  # Primera detección: (17, 3)
                
                # Aplanar los keypoints: [x1,y1,c1, x2,y2,c2, ...]
                frame_kpts[0::3] = kpts[:, 0]  # x
                frame_kpts[1::3] = kpts[:, 1]  # y
                frame_kpts[2::3] = kpts[:, 2]  # confidence

        keypoints_sequence.append(frame_kpts)
        # if pbar_frames: pbar_frames.update(1)

    cap.release()
    # if pbar_frames: pbar_frames.close()

    if not keypoints_sequence:
        print(f"Advertencia: No se extrajeron keypoints de {video_path}")
        return None

    return np.array(keypoints_sequence)  # Forma (T, 51) donde T es el número de frames

def find_all_videos(root_dir):
    """
    Busca recursivamente todos los archivos de vídeo en root_dir y subdirectorios.
    Retorna una lista de rutas relativas a root_dir.
    """
    video_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                # Obtener ruta completa
                abs_path = os.path.join(root, file)
                video_files.append(abs_path)
    return video_files

def main(args):
    # Crear directorios de salida
    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.val_dir, exist_ok=True)

    # Cargar modelo YOLO-Pose
    print(f"Cargando modelo YOLO-Pose desde {args.model_path}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    try:
        # Asegúrate que el modelo existe. Si no, Ultralytics intentará descargarlo.
        model = YOLO(args.model_path)
        model.to(device) # Mover el modelo al dispositivo
    except Exception as e:
        print(f"Error al cargar el modelo YOLO ({args.model_path}): {e}")
        print("Verifica la ruta y el nombre del modelo. Si usas un modelo custom, asegúrate que el archivo .pt existe.")
        print("Si es un modelo estándar de Ultralytics (ej: 'yolov8n-pose.pt'), asegúrate de tener conexión a internet la primera vez.")
        return

    # Inicializar CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Listar vídeos recursivamente
    try:
        print(f"Buscando videos en {args.raw_dir} y subdirectorios...")
        all_video_paths = find_all_videos(args.raw_dir)
    except FileNotFoundError:
        print(f"Error: El directorio de vídeos raw '{args.raw_dir}' no existe.")
        print("Por favor, crea el directorio y descarga/coloca los vídeos del dataset de Kaggle allí.")
        return

    if not all_video_paths:
        print(f"Error: No se encontraron archivos de vídeo en {args.raw_dir} ni sus subdirectorios")
        print("Asegúrate de que los vídeos del 'Hand Wash Dataset' están en esa estructura de carpetas.")
        return

    print(f"Encontrados {len(all_video_paths)} vídeos en {args.raw_dir} y subdirectorios")

    # Filtrar vídeos para los que se puede obtener etiqueta
    video_paths_with_labels = []
    for path in all_video_paths:
        label = get_label_from_filename(path)
        if label != -1:
            video_paths_with_labels.append((path, label))
        else:
            print(f"Omitiendo vídeo {path} por no poder determinar etiqueta.")

    if not video_paths_with_labels:
        print("Error: No se pudo asignar etiqueta a ningún vídeo. Revisa la función 'get_label_from_filename'.")
        return

    print(f"Se procesarán {len(video_paths_with_labels)} vídeos con etiquetas válidas.")

    # Mezclar y dividir
    random.shuffle(video_paths_with_labels)
    split_index = int(len(video_paths_with_labels) * args.split_ratio)
    train_paths_labels = video_paths_with_labels[:split_index]
    val_paths_labels = video_paths_with_labels[split_index:]

    print(f"Dividiendo en {len(train_paths_labels)} para entrenamiento y {len(val_paths_labels)} para validación.")

    datasets = {
        'train': {'paths_labels': train_paths_labels, 'dir': args.train_dir, 'labels_file': os.path.join(args.train_dir, 'labels.csv')},
        'val': {'paths_labels': val_paths_labels, 'dir': args.val_dir, 'labels_file': os.path.join(args.val_dir, 'labels.csv')}
    }

    for split_name, data in datasets.items():
        print(f"\nProcesando vídeos para {split_name}...")
        output_dir = data['dir']
        labels_list = []

        # Usar tqdm para la barra de progreso general
        for video_path, label in tqdm(data['paths_labels'], desc=f"Extrayendo keypoints ({split_name})"):
            # Generar un nombre de archivo único basado en la ruta relativa
            # Usar el nombre base del video original para evitar colisiones si hay nombres duplicados en diferentes subcarpetas de pasos
            video_basename = os.path.basename(video_path)
            npy_filename = os.path.splitext(video_basename)[0] + '.npy'
            # Asegurarse de que el nombre es único si hay videos con el mismo nombre en diferentes carpetas de pasos
            # (Aunque el DataFrame de etiquetas lo manejará correctamente)

            keypoints = extract_keypoints_from_video(video_path, model, clahe)

            if keypoints is not None and keypoints.shape[0] > 0: # Asegurarse que hay frames con keypoints
                save_path = os.path.join(output_dir, npy_filename)
                try:
                    np.save(save_path, keypoints)
                    labels_list.append({'filename': npy_filename, 'label': label})
                except Exception as e:
                    print(f"Error al guardar {save_path}: {e}")
            else:
                print(f"Omitiendo {video_path} (no se pudieron extraer keypoints válidos)." )

        # Guardar archivo de etiquetas para este split
        if labels_list:
            labels_df = pd.DataFrame(labels_list)
            labels_df.to_csv(data['labels_file'], index=False)
            print(f"Archivo de etiquetas de {split_name} guardado en {data['labels_file']} con {len(labels_list)} muestras")
        else:
            print(f"Advertencia: No se generaron datos de {split_name}. Verifica los vídeos y el proceso de extracción.")

    print("\nExtracción de keypoints completada.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extrae keypoints de vídeos de lavado de manos usando YOLO-Pose.')
    # Use the constants defined above for defaults
    parser.add_argument('--raw-dir', type=str, default=RAW_VIDEOS_DIR,
                        help=f'Directorio con los vídeos originales (default: {RAW_VIDEOS_DIR})')
    parser.add_argument('--train-dir', type=str, default=KEYPOINTS_TRAIN_DIR,
                        help=f'Directorio para guardar keypoints de entrenamiento (default: {KEYPOINTS_TRAIN_DIR})')
    parser.add_argument('--val-dir', type=str, default=KEYPOINTS_VAL_DIR,
                        help=f'Directorio para guardar keypoints de validación (default: {KEYPOINTS_VAL_DIR})')
    parser.add_argument('--model-path', type=str, default=DEFAULT_MODEL_PATH,
                        help=f'Ruta al modelo YOLO Pose (ej: yolov8n-pose.pt o tu modelo custom .pt) (default: {DEFAULT_MODEL_PATH})')
    parser.add_argument('--split-ratio', type=float, default=DEFAULT_SPLIT_RATIO,
                        help=f'Ratio para dividir datos en entrenamiento/validación (default: {DEFAULT_SPLIT_RATIO})')

    args = parser.parse_args()

    # Validar que el modelo especificado sea plausible (termina en .pt)
    if not args.model_path.endswith('.pt'):
        print(f"Advertencia: El nombre del modelo '{args.model_path}' no termina en .pt. Asegúrate de que es correcto.")

    main(args)
