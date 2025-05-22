import cv2
import numpy as np
import torch
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import time
import argparse
import os
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# --- Configuración --- 
DEFAULT_YOLO_MODEL = 'yolo11n-pose-hands.pt' # Modelo YOLO afinado para manos
WEBCAM_INDEX = 0 # Índice de la cámara USB (puede variar)
TARGET_FPS = 30
MIN_SOAP_SECONDS = 20 # Duración mínima requerida para la fase de enjabonado

# MQTT Config
MQTT_BROKER = "localhost" # Cambiar por la IP/hostname de tu broker MQTT
MQTT_PORT = 1883
MQTT_TOPIC_STATUS = "handwash/status" # Tópico para publicar OK/ALERTA

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

    return None, results

def draw_keypoints_on_frame(frame, results, focus_on_hands=True):
    """Dibuja los keypoints en el frame, enfocándose en las manos si se especifica."""
    if results and len(results) > 0:
        annotated_frame = results[0].plot()
        return annotated_frame
    return frame

# --- Función para detectar ambas manos ---
def both_hands_present(yolo_results, conf_thresh=0.5):
    """
    Returns True if two hands are detected with confidence above threshold.
    Assumes YOLO pose model outputs one detection per hand.
    """
    if yolo_results and len(yolo_results) > 0 and yolo_results[0].keypoints is not None:
        kpts_data = yolo_results[0].keypoints.data
        # Each detection is a hand, so need at least 2
        if kpts_data.shape[0] >= 2:
            # Optionally, check average confidence for each hand
            hand1_conf = np.mean(kpts_data[0][:,2].cpu().numpy())
            hand2_conf = np.mean(kpts_data[1][:,2].cpu().numpy())
            return hand1_conf > conf_thresh and hand2_conf > conf_thresh
    return False

# --- Función Principal --- 
def main(args):
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # --- Cargar Modelo YOLO --- 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.yolo_model):
        yolo_model_path_abs = os.path.join(script_dir, args.yolo_model)
    else:
        yolo_model_path_abs = args.yolo_model

    if not os.path.exists(yolo_model_path_abs):
        print(f"Error: No se encontró el modelo YOLO en '{yolo_model_path_abs}' (original path: '{args.yolo_model}')")
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

    # Timer variables
    both_hands_timer = 0.0
    last_frame_time = time.time()
    hands_visible = False
    alert_sent = False
    status = "OK"

    # Obtener dimensiones del vídeo para el layout
    ret, temp_frame = cap.read()
    if not ret:
        print("Error al leer el primer frame. Verificar webcam.")
        return
    height, width = temp_frame.shape[:2]

    print("Iniciando bucle de inferencia...")
    try:
        while True:
            frame_start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Error al leer frame de la webcam.")
                break

            # Extraer keypoints y obtener resultados del modelo YOLO
            _, yolo_results = extract_keypoints_frame(frame, yolo_model, clahe)
            hands_present = both_hands_present(yolo_results)

            # Timer logic
            now = time.time()
            elapsed = now - last_frame_time
            last_frame_time = now
            if hands_present:
                both_hands_timer += elapsed
                hands_visible = True
            else:
                hands_visible = False

            # Estado y alerta
            if both_hands_timer >= MIN_SOAP_SECONDS:
                status = "OK"
                if alert_sent:
                    publish_mqtt(mqtt_client, MQTT_TOPIC_STATUS, status)
                    alert_sent = False
            else:
                if not alert_sent and both_hands_timer > 0:
                    status = "ALERTA_DURACION_INSUFICIENTE"
                    publish_mqtt(mqtt_client, MQTT_TOPIC_STATUS, status)
                    alert_sent = True

            # Mostrar video
            if args.show_video:
                annotated_frame = draw_keypoints_on_frame(frame, yolo_results, focus_on_hands=True)
                display_img = annotated_frame.copy()
                # Mostrar timer
                cv2.putText(display_img, f"Tiempo ambas manos: {both_hands_timer:.1f}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if status=="OK" else (0,0,255), 2)
                # Mostrar alerta
                if status != "OK":
                    cv2.putText(display_img, "ALERTA: ENJABONADO CORTO", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow('Handwash Monitor', display_img)
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
    parser.add_argument('--yolo-model', type=str, default=DEFAULT_YOLO_MODEL, help=f'Ruta o nombre del modelo YOLO-Pose (default: {DEFAULT_YOLO_MODEL})')
    parser.add_argument('--webcam-index', type=int, default=WEBCAM_INDEX, help=f'Índice de la cámara a usar (default: {WEBCAM_INDEX})')
    parser.add_argument('--mqtt-broker', type=str, default=MQTT_BROKER, help=f'Dirección del broker MQTT (default: {MQTT_BROKER})')
    parser.add_argument('--mqtt-port', type=int, default=MQTT_PORT, help=f'Puerto del broker MQTT (default: {MQTT_PORT})')
    parser.add_argument('--show-video', action='store_true', help='Mostrar la ventana de vídeo con la fase detectada.')
    args = parser.parse_args()
    # Establecer show_video a True por defecto, ya que queremos ver los keypoints
    args.show_video = True
    main(args)
