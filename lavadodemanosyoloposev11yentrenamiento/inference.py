import cv2
import numpy as np
import torch
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import time
import argparse
import os
from collections import deque

# Importar clases necesarias
from model.tsm_gru import TSM_GRU

# --- Configuración --- 
DEFAULT_MODEL_PATH = 'best_tsm_gru.pth'
DEFAULT_YOLO_MODEL = 'YOLO11n-pose.pt' # Modelo YOLO para extracción de keypoints (in the same directory)
WEBCAM_INDEX = 0 # Índice de la cámara USB (puede variar)
TARGET_FPS = 30
WINDOW_SECONDS = 5 # Duración de la ventana deslizante en segundos
SEQUENCE_LENGTH = TARGET_FPS * WINDOW_SECONDS # 150 frames
MIN_SOAP_SECONDS = 20 # Duración mínima requerida para la fase de enjabonado

# Número de keypoints que produce el modelo YOLOv8 pose (17 para cuerpo completo)
YOLOV8_POSE_KEYPOINTS = 17
FEATURE_DIM = YOLOV8_POSE_KEYPOINTS * 3  # 51: x, y, confidence para cada keypoint

# MQTT Config
MQTT_BROKER = "localhost" # Cambiar por la IP/hostname de tu broker MQTT
MQTT_PORT = 1883
MQTT_TOPIC_STATUS = "handwash/status" # Tópico para publicar OK/ALERTA
MQTT_TOPIC_PHASE = "handwash/phase"  # Tópico para publicar la fase detectada

# Mapeo de etiquetas a nombres y colores (para visualización)
LABEL_NAMES = {0: 'Mojar', 1: 'Enjabonado', 2: 'Frotado', 3: 'Aclarado', -1: 'Desconocido'}
LABEL_COLORS = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0), -1: (128, 128, 128)}
SOAP_LABEL_INDEX = 1 # Índice de la etiqueta "Enjabonado"

# --- Funciones Auxiliares --- 
def connect_mqtt(broker, port):
    """Conecta al broker MQTT."""
    client = mqtt.Client()
    try:
        client.connect(broker, port, 60)
        client.loop_start() # Inicia el bucle de red en segundo plano
        print(f"Conectado al broker MQTT en {broker}:{port}")
        return client
    except Exception as e:
        print(f"Error al conectar al broker MQTT: {e}")
        return None

def publish_mqtt(client, topic, message):
    """Publica un mensaje en un tópico MQTT."""
    if client and client.is_connected():
        result = client.publish(topic, message)
        return result.rc == mqtt.MQTT_ERR_SUCCESS
    else:
        return False

def extract_keypoints_frame(frame, yolo_model, clahe):
    """Extrae keypoints de un solo frame usando YOLOv8 pose."""
    # Preprocesado CLAHE
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe_frame_gray = clahe.apply(gray)
    clahe_frame_bgr = cv2.cvtColor(clahe_frame_gray, cv2.COLOR_GRAY2BGR)

    # Inferencia YOLO-Pose
    results = yolo_model(clahe_frame_bgr, verbose=False)

    # Inicializar vector de características (51 = 17 keypoints * 3)
    frame_kpts = np.zeros(FEATURE_DIM)

    if results and len(results) > 0 and results[0].keypoints is not None and results[0].keypoints.shape[1] > 0:
        # Obtener los datos de keypoints
        kpts_data = results[0].keypoints.data
            
        # Seleccionar la persona principal (la primera detección)
        if kpts_data.shape[0] > 0:
            kpts = kpts_data[0].cpu().numpy()  # Primera detección: (17, 3)
                
            # Aplanar los keypoints: [x1,y1,c1, x2,y2,c2, ...]
            frame_kpts[0::3] = kpts[:, 0]  # x
            frame_kpts[1::3] = kpts[:, 1]  # y
            frame_kpts[2::3] = kpts[:, 2]  # confidence

    return frame_kpts

# --- Clase para manejar el estado del lavado --- 
class HandwashMonitor:
    def __init__(self, min_soap_duration_frames, soap_label_index):
        self.min_soap_duration_frames = min_soap_duration_frames
        self.soap_label_index = soap_label_index
        self.current_phase = -1
        self.phase_start_time = None
        self.last_alert_time = 0
        self.alert_cooldown = 10 # Segundos de cooldown para no spamear alertas
        self.alert_active = False
        self.max_soap_duration_achieved = 0 # Máxima duración continua de enjabonado vista

    def update_phase(self, predicted_phase):
        current_time = time.time()
        alert_status = "OK"

        if predicted_phase != self.current_phase:
            # Cambio de fase
            if self.current_phase == self.soap_label_index:
                # Si estábamos enjabonando, verificar duración
                duration = current_time - self.phase_start_time
                self.max_soap_duration_achieved = max(self.max_soap_duration_achieved, duration)
                print(f"Fase 'Enjabonado' terminada. Duración: {duration:.1f}s. Máxima duración vista: {self.max_soap_duration_achieved:.1f}s")
                # Resetear alerta si estaba activa por duración insuficiente
                if self.alert_active:
                     print("Alerta de duración desactivada.")
                     self.alert_active = False
                     alert_status = "OK" # Enviar OK al salir de enjabonado corto

            # Actualizar fase y tiempo de inicio
            print(f"Cambio de fase: {LABEL_NAMES.get(self.current_phase, 'N/A')} -> {LABEL_NAMES.get(predicted_phase, 'N/A')}")
            self.current_phase = predicted_phase
            self.phase_start_time = current_time
            # Si la nueva fase NO es enjabonado, resetear la duración máxima vista
            if self.current_phase != self.soap_label_index:
                self.max_soap_duration_achieved = 0

        elif predicted_phase == self.soap_label_index:
            # Seguimos en la fase de enjabonado
            duration = current_time - self.phase_start_time
            self.max_soap_duration_achieved = max(self.max_soap_duration_achieved, duration)
            # Verificar si la duración actual es suficiente
            if duration < self.min_soap_duration_frames / TARGET_FPS:
                # Aún no es suficiente, comprobar si emitir alerta
                if not self.alert_active and (current_time - self.last_alert_time > self.alert_cooldown):
                    pass
            else:
                # Duración suficiente, si la alerta estaba activa, desactivarla
                if self.alert_active:
                    print("Duración mínima de enjabonado alcanzada. Alerta desactivada.")
                    self.alert_active = False
                    alert_status = "OK"

        if predicted_phase != self.soap_label_index and self.current_phase == self.soap_label_index:
             if self.max_soap_duration_achieved < self.min_soap_duration_frames / TARGET_FPS:
                 if current_time - self.last_alert_time > self.alert_cooldown:
                     print(f"¡ALERTA! La fase de enjabonado duró solo {self.max_soap_duration_achieved:.1f}s (requerido: {self.min_soap_duration_frames / TARGET_FPS:.1f}s)")
                     alert_status = "ALERTA_DURACION_INSUFICIENTE"
                     self.alert_active = True
                     self.last_alert_time = current_time

        return alert_status, self.current_phase

# --- Función Principal --- 
def main(args):
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # --- Construct absolute paths for models --- 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # TSM-GRU model path
    if not os.path.isabs(args.model_path):
        tsm_model_path_abs = os.path.join(script_dir, args.model_path)
    else:
        tsm_model_path_abs = args.model_path

    # YOLO model path
    if not os.path.isabs(args.yolo_model):
        yolo_model_path_abs = os.path.join(script_dir, args.yolo_model)
    else:
        yolo_model_path_abs = args.yolo_model

    # --- Cargar Modelo TSM-GRU --- 
    if not os.path.exists(tsm_model_path_abs):
        print(f"Error: No se encontró el modelo TSM-GRU en '{tsm_model_path_abs}' (original path: '{args.model_path}')")
        return
    print(f"Cargando modelo TSM-GRU desde {tsm_model_path_abs}...")
    try:
        checkpoint = torch.load(tsm_model_path_abs, map_location=device)
        model_args = checkpoint.get('args', None)
        if model_args:
            input_dim = model_args.get('input_dim', FEATURE_DIM)
            if input_dim != FEATURE_DIM:
                print(f"Advertencia: El modelo fue entrenado con input_dim={input_dim}, pero se esperaba {FEATURE_DIM}")
                print("Esto puede ocurrir si se cambió de un modelo YOLO custom a YOLOv8 pose estándar.")
                input_dim = FEATURE_DIM
                
            tsm_gru_model = TSM_GRU(input_dim=input_dim,
                                    hidden_dim=model_args.get('hidden_dim', 256), # Default 256
                                    num_classes=model_args.get('num_classes', 4),
                                    num_layers_gru=model_args.get('gru_layers', 2), # Default 2
                                    tsm_segments=model_args.get('tsm_segments', 8),
                                    tsm_shift_div=model_args.get('tsm_shift_div', 3), # Default 3
                                    bidirectional=model_args.get('bidirectional', False),
                                    use_attention=model_args.get('use_attention', False),
                                    dropout=0.0).to(device)
                                    
            try:
                tsm_gru_model.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e:
                print(f"Error al cargar el modelo: {e}")
                print("El modelo guardado tiene una arquitectura incompatible. Deberás reentrenar el modelo con la nueva dimensión de entrada.")
                return
                
            global SEQUENCE_LENGTH
            train_seq_len = model_args.get('seq_len', SEQUENCE_LENGTH)
            if train_seq_len != SEQUENCE_LENGTH:
                print(f"Advertencia: La longitud de secuencia de inferencia ({SEQUENCE_LENGTH}) difiere de la de entrenamiento ({train_seq_len}). Ajustando a {train_seq_len}.")
                SEQUENCE_LENGTH = train_seq_len
        else:
            print("Advertencia: No se encontraron args en checkpoint. Usando defaults para TSM_GRU con dimensión de entrada 51.")
            # Ensure these defaults match evaluate.py fallbacks
            tsm_gru_model = TSM_GRU(input_dim=FEATURE_DIM,
                                    hidden_dim=256, # Match evaluate.py fallback
                                    num_classes=4,
                                    num_layers_gru=2, # Match evaluate.py fallback
                                    tsm_segments=8,
                                    tsm_shift_div=3, # Match evaluate.py fallback
                                    bidirectional=False, # Default to False if no args
                                    use_attention=False, # Default to False if no args
                                    dropout=0.0).to(device)
            try:
                tsm_gru_model.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e:
                print(f"Error al cargar el modelo: {e}")
                print("El modelo guardado tiene una arquitectura incompatible. Deberás reentrenar el modelo con la nueva dimensión de entrada.")
                return
        
        tsm_gru_model.eval()
        print("Modelo TSM-GRU cargado.")
    except Exception as e:
        print(f"Error al cargar el modelo TSM-GRU: {e}")
        return

    # --- Cargar Modelo YOLO --- 
    if not os.path.exists(yolo_model_path_abs):
        print(f"Error: No se encontró el modelo YOLO en '{yolo_model_path_abs}' (original path: '{args.yolo_model}')")
        print("Asegúrate de que el archivo .pt está en el mismo directorio que inference.py o que la ruta es correcta.")
        return
    print(f"Cargando modelo YOLO-Pose desde {yolo_model_path_abs}...")
    try:
        yolo_model = YOLO(yolo_model_path_abs)
        yolo_model.to(device)
        print("Modelo YOLO-Pose cargado.")
    except Exception as e:
        print(f"Error al cargar el modelo YOLO: {e}")
        return

    # --- Inicializar Captura de Vídeo --- 
    print(f"Iniciando captura desde webcam (índice: {args.webcam_index})...")
    cap = cv2.VideoCapture(args.webcam_index)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la webcam {args.webcam_index}.")
        return
    
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Webcam abierta. FPS solicitado: {TARGET_FPS}, FPS real: {actual_fps:.2f}")
    frame_delay = 1.0 / TARGET_FPS

    # --- Inicializar MQTT --- 
    mqtt_client = connect_mqtt(args.mqtt_broker, args.mqtt_port)

    # --- Inicializar otros componentes --- 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    keypoints_buffer = deque(maxlen=SEQUENCE_LENGTH)
    monitor = HandwashMonitor(min_soap_duration_frames=MIN_SOAP_SECONDS * TARGET_FPS, 
                              soap_label_index=SOAP_LABEL_INDEX)
    last_phase_prediction = -1

    print("Iniciando bucle de inferencia...")
    try:
        while True:
            frame_start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Error al leer frame de la webcam.")
                break

            current_keypoints = extract_keypoints_frame(frame, yolo_model, clahe)
            keypoints_buffer.append(current_keypoints)

            predicted_phase_index = -1
            if len(keypoints_buffer) == SEQUENCE_LENGTH:
                sequence_np = np.array(keypoints_buffer)
                sequence_tensor = torch.tensor(sequence_np, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = tsm_gru_model(sequence_tensor)
                    _, predicted = torch.max(outputs.data, 1)
                    predicted_phase_index = predicted.item()
                
                alert_status, current_phase_index = monitor.update_phase(predicted_phase_index)
                last_phase_prediction = current_phase_index

                if alert_status != "OK":
                    publish_mqtt(mqtt_client, MQTT_TOPIC_STATUS, alert_status)
                publish_mqtt(mqtt_client, MQTT_TOPIC_PHASE, LABEL_NAMES.get(last_phase_prediction, "Desconocido"))
            
            else:
                current_phase_index = last_phase_prediction

            if args.show_video:
                if np.sum(current_keypoints) > 0:
                    for i in range(0, FEATURE_DIM, 3):
                        x, y, conf = current_keypoints[i:i+3]
                        if conf > 0.2:
                            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

                phase_name = LABEL_NAMES.get(current_phase_index, "Iniciando...")
                color = LABEL_COLORS.get(current_phase_index, (255, 255, 255))
                cv2.putText(frame, f"Fase: {phase_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                
                if monitor.alert_active:
                     cv2.putText(frame, "ALERTA: ENJABONADO CORTO", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.imshow('Handwash Inference', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            elapsed_time = time.time() - frame_start_time
            sleep_time = frame_delay - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        print("Deteniendo...")
        cap.release()
        if args.show_video:
            cv2.destroyAllWindows()
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
            print("Cliente MQTT desconectado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inferencia en tiempo real para supervisión de lavado de manos.')
    parser.add_argument('--model-path', type=str, default=DEFAULT_MODEL_PATH, help=f'Ruta al modelo TSM-GRU entrenado (.pth) (default: {DEFAULT_MODEL_PATH})')
    parser.add_argument('--yolo-model', type=str, default=DEFAULT_YOLO_MODEL, help=f'Ruta o nombre del modelo YOLO-Pose (default: {DEFAULT_YOLO_MODEL})')
    parser.add_argument('--webcam-index', type=int, default=WEBCAM_INDEX, help=f'Índice de la cámara a usar (default: {WEBCAM_INDEX})')
    parser.add_argument('--mqtt-broker', type=str, default=MQTT_BROKER, help=f'Dirección del broker MQTT (default: {MQTT_BROKER})')
    parser.add_argument('--mqtt-port', type=int, default=MQTT_PORT, help=f'Puerto del broker MQTT (default: {MQTT_PORT})')
    parser.add_argument('--show-video', action='store_true', help='Mostrar la ventana de vídeo con la fase detectada.')
    # Add fallback arguments for model architecture (used if not in checkpoint)
    parser.add_argument('--bidirectional', action='store_true', help='Usar GRU bidireccional (fallback/override).')
    parser.add_argument('--use-attention', action='store_true', help='Usar mecanismo de atención (fallback/override).')

    args = parser.parse_args()
    main(args)
