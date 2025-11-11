"""
Sistema de Videovigilancia Inteligente para Detección de Actividades Anómalas
Basado en el artículo de Sathiyavathi et al. (2021)
Integra CNN, YOLOv8-pose, clasificación de riesgo y API REST
"""

import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os

# ============================================================================
# CONFIGURACIÓN Y CARGA DE MODELOS
# ============================================================================

print("=" * 70)
print("SISTEMA DE VIDEOVIGILANCIA INTELIGENTE")
print("Detección de Actividades Anómalas usando CNN")
print("Basado en Sathiyavathi et al. (2021)")
print("=" * 70)

# Cargar modelos YOLOv8
print("\nCargando modelos...")
model_pose = YOLO("yolov8n-pose.pt")
print("Modelo de pose cargado")

# Modelo para detección general de objetos
model_objects = YOLO("yolov8n.pt")
print("Modelo de objetos cargado")

# Modelo entrenado para detección de armas (prioritario)
model_weapons = None

# Intentar cargar desde Roboflow primero
try:
    from roboflow import Roboflow
    roboflow_api_key = os.getenv("ROBOFLOW_API_KEY", "")
    if roboflow_api_key:
        print("Intentando cargar modelo desde Roboflow...")
        rf = Roboflow(api_key=roboflow_api_key)
        project = rf.workspace("my-first-project-lchlk").project("1")
        model_weapons = project.version(1).model
        print("Modelo de armas cargado desde Roboflow (my-first-project-lchlk/1)")
        print("   Nota: Configura ROBOFLOW_API_KEY como variable de entorno si aún no lo has hecho")
    else:
        print("ROBOFLOW_API_KEY no configurada, intentando modelo local...")
        raise ValueError("API key no configurada")
except Exception as e:
    print(f"No se pudo cargar desde Roboflow: {e}")
    print("Intentando cargar modelo local...")
    
    # Si falla Roboflow, intentar cargar modelo local
    weapons_model_paths = [
        "best.pt",  # Raíz del proyecto
    ]
    
    # Buscar en runs/ también
    try:
        import glob
        runs_models = glob.glob("runs/**/weights/best.pt", recursive=True)
        weapons_model_paths.extend(runs_models[:3])  # Agregar hasta 3 modelos encontrados
    except:
        pass
    
    for path in weapons_model_paths:
        if os.path.exists(path):
            try:
                model_weapons = YOLO(path)
                print(f"Modelo de armas entrenado cargado desde: {path}")
                break
            except Exception as e:
                print(f"Advertencia: Error al cargar {path}: {e}")
                continue

if model_weapons is None:
    print("Advertencia: No se encontró modelo de armas entrenado")
    print("   Usando detección con modelo general (knife, scissors)")
    print("   Para usar Roboflow, configura la variable de entorno ROBOFLOW_API_KEY")
    print("   O coloca tu modelo como 'best.pt' en la raíz del proyecto")

# Clases que se consideran armas en el modelo general (como respaldo)
WEAPON_CLASSES = ['knife', 'scissors']  # YOLOv8n puede detectar estos objetos

print("\nSistema listo\n")

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def calculate_distance(box1, box2):
    """Calcula distancia entre centros de dos bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
    center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def calculate_iou(box1, box2):
    """Calcula Intersection over Union (IoU) entre dos bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

def draw_simple_skeleton(frame, keypoints):
    """Dibuja esqueleto simplificado de la persona"""
    if keypoints is None or len(keypoints) == 0:
        return frame
    
    try:
        # Conexiones principales del esqueleto
        connections = [
            [5, 7], [7, 9],   # Brazo izquierdo
            [6, 8], [8, 10],  # Brazo derecho
            [11, 13], [13, 15],  # Pierna izquierda
            [12, 14], [14, 16],  # Pierna derecha
            [5, 6], [11, 12], [5, 11], [6, 12]  # Torso
        ]
        
        frame_h, frame_w = frame.shape[:2]
        
        # Dibujar conexiones
        for conn in connections:
            if conn[0] < len(keypoints) and conn[1] < len(keypoints):
                pt1 = keypoints[conn[0]]
                pt2 = keypoints[conn[1]]
                
                if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                    pt1_int = (int(pt1[0]), int(pt1[1]))
                    pt2_int = (int(pt2[0]), int(pt2[1]))
                    
                    if (0 <= pt1_int[0] < frame_w and 0 <= pt1_int[1] < frame_h and
                        0 <= pt2_int[0] < frame_w and 0 <= pt2_int[1] < frame_h):
                        cv2.line(frame, pt1_int, pt2_int, (0, 255, 255), 2)
        
        # Dibujar puntos clave principales
        for i, kpt in enumerate(keypoints):
            if kpt[0] > 0 and kpt[1] > 0:
                x, y = int(kpt[0]), int(kpt[1])
                if 0 <= x < frame_w and 0 <= y < frame_h:
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
    except:
        pass
    
    return frame

# ============================================================================
# FUNCIÓN PRINCIPAL DE DETECCIÓN
# ============================================================================

def detect_and_classify(frame, model_pose, model_objects, model_weapons=None, weapon_classes=None):
    """Detecta personas, objetos y armas, genera alertas"""
    weapons_detected = []
    objects_detected = []
    num_people = 0
    
    try:
        # Áreas donde ya se detectaron armas (para evitar duplicados)
        weapon_areas = []
        
        # 1. Detección de armas con modelo entrenado (prioritario) - SIEMPRE ROJO
        if model_weapons is not None:
            try:
                # Para modelos de Roboflow, usar predict() con formato JSON
                if hasattr(model_weapons, 'predict') and not hasattr(model_weapons, 'names'):
                    # Modelo de Roboflow - guardar frame temporalmente y predecir
                    import tempfile
                    temp_path = tempfile.mktemp(suffix='.jpg')
                    cv2.imwrite(temp_path, frame)
                    
                    try:
                        predictions_json = model_weapons.predict(temp_path, confidence=10, overlap=30).json()
                        if 'predictions' in predictions_json:
                            for prediction in predictions_json['predictions']:
                                x = prediction.get('x', 0)
                                y = prediction.get('y', 0)
                                width = prediction.get('width', 0)
                                height = prediction.get('height', 0)
                                confidence = float(prediction.get('confidence', 0))
                                label = prediction.get('class', 'weapon')
                                
                                x1 = int(x - width / 2)
                                y1 = int(y - height / 2)
                                x2 = int(x + width / 2)
                                y2 = int(y + height / 2)
                                
                                if confidence > 0.1:
                                    bbox = (x1, y1, x2, y2)
                                    weapons_detected.append((bbox, label, confidence))
                                    weapon_areas.append(bbox)
                                    
                                    # Dibujar arma detectada (ROJO)
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                                    text = f"ARMA: {label.upper()} {confidence:.2f}"
                                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                                    cv2.rectangle(frame, (x1-2, y1-25), (x1 + text_size[0] + 5, y1-5), (0, 0, 0), -1)
                                    cv2.putText(frame, text, (x1, y1 - 10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                else:
                    # Modelo YOLO estándar
                    results_weapons = model_weapons(frame, verbose=False, conf=0.1, imgsz=640)
                    for r in results_weapons:
                        if r.boxes is not None and len(r.boxes) > 0:
                            for box in r.boxes:
                                confidence = float(box.conf[0])
                                if confidence > 0.1:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    class_id = int(box.cls[0])
                                    label = model_weapons.names[class_id]
                                    
                                    bbox = (x1, y1, x2, y2)
                                    weapons_detected.append((bbox, label, confidence))
                                    weapon_areas.append(bbox)
                                    
                                    # Dibujar arma detectada (ROJO)
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                                    text = f"ARMA: {label.upper()} {confidence:.2f}"
                                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                                    cv2.rectangle(frame, (x1-2, y1-25), (x1 + text_size[0] + 5, y1-5), (0, 0, 0), -1)
                                    cv2.putText(frame, text, (x1, y1 - 10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            except Exception as e:
                print(f"Error en detección de armas: {e}")
        
        # 2. Detección de objetos generales (excluir áreas donde ya hay armas detectadas)
        results_objects = model_objects(frame, verbose=False, conf=0.25, imgsz=640)
        for r in results_objects:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    confidence = float(box.conf[0])
                    if confidence > 0.25:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        class_id = int(box.cls[0])
                        label = model_objects.names[class_id]
                        
                        # Filtrar personas (ya las detectamos con pose)
                        if label == 'person':
                            continue
                        
                        # Verificar si esta área ya fue detectada como arma
                        is_weapon_area = False
                        for w_bbox in weapon_areas:
                            iou = calculate_iou((x1, y1, x2, y2), w_bbox)
                            if iou > 0.2:  # Si hay superposición, es un área de arma
                                is_weapon_area = True
                                break
                        
                        if is_weapon_area:
                            continue  # Saltar, ya está marcado como arma
                        
                        # Verificar si es un arma del modelo general (solo si no hay modelo entrenado)
                        is_weapon = False
                        if model_weapons is None and weapon_classes and label.lower() in weapon_classes:
                            is_weapon = True
                        
                        if is_weapon:
                            # Es un arma detectada por modelo general (rojo)
                            weapons_detected.append(((x1, y1, x2, y2), label, confidence))
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                            text = f"ARMA: {label.upper()} {confidence:.2f}"
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            cv2.rectangle(frame, (x1-2, y1-25), (x1 + text_size[0] + 5, y1-5), (0, 0, 0), -1)
                            cv2.putText(frame, text, (x1, y1 - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            # Es un objeto normal (azul)
                            objects_detected.append(((x1, y1, x2, y2), label, confidence))
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            text = f"{label.upper()} {confidence:.2f}"
                            cv2.putText(frame, text, (x1, y1 - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # 3. Detección de personas con pose
        results_pose = model_pose(frame, verbose=False, conf=0.5, imgsz=416)
        
        for r in results_pose:
            if r.keypoints is not None and r.boxes is not None:
                num_people = len(r.boxes)
                
                for i in range(len(r.boxes)):
                    try:
                        box = r.boxes[i]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bbox = (x1, y1, x2, y2)
                        
                        # Obtener keypoints
                        kpts = r.keypoints.data[i]
                        if hasattr(kpts, 'cpu'):
                            keypoints = kpts.cpu().numpy()[:, :2]
                        else:
                            keypoints = np.array(kpts)[:, :2]
                        
                        # Verificar si tiene arma cerca
                        has_weapon = False
                        person_size = np.sqrt((x2 - x1) * (y2 - y1))
                        for weapon_bbox, weapon_label, weapon_conf in weapons_detected:
                            distance = calculate_distance(bbox, weapon_bbox)
                            if distance < person_size * 0.5 or distance < 100:
                                has_weapon = True
                                break
                        
                        # Dibujar esqueleto
                        frame = draw_simple_skeleton(frame, keypoints)
                        
                        # Dibujar bbox según si tiene arma
                        if has_weapon:
                            # Rojo si tiene arma
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            text = f"PERSONA {i+1} - ARMA DETECTADA"
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_size[0] + 5, y1), (0, 0, 0), -1)
                            cv2.putText(frame, text, (x1, y1 - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        else:
                            # Verde si no tiene arma
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"Persona {i+1}", (x1, y1 - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    except:
                        continue
        
        # Mostrar estadísticas en pantalla (con fondo para mejor visibilidad)
        stats_y = 30
        stats_spacing = 30
        
        # Fondo para estadísticas
        cv2.rectangle(frame, (5, 5), (350, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (350, 100), (255, 255, 255), 2)
        
        # Personas detectadas
        cv2.putText(frame, f"Personas: {num_people}", (10, stats_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Objetos detectados
        if objects_detected:
            cv2.putText(frame, f"Objetos: {len(objects_detected)}", (10, stats_y + stats_spacing), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Armas detectadas
        if weapons_detected:
            cv2.putText(frame, f"Armas: {len(weapons_detected)}", (10, stats_y + stats_spacing * 2), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    except:
        pass
    
    return frame, num_people, weapons_detected, objects_detected

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    print("Opciones:")
    print("  - Presiona Enter para usar cámara web")
    print("  - O ingresa la ruta a un archivo de video")
    
    source = input("\nFuente de video [Enter para cámara]: ").strip()
    
    cap = cv2.VideoCapture(0 if source == "" else source)
    if not cap.isOpened():
        print("Error: No se pudo abrir la fuente de video")
        return
    
    print("\nSistema activo. Presiona 'q' para salir\n")
    
    frame_number = 0
    skip_frames = 2
    last_frame = None
    last_weapon_alert_time = {}
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            
            if frame_number % skip_frames == 0:
                frame, num_people, weapons_detected, objects_detected = detect_and_classify(
                    frame, model_pose, model_objects, model_weapons, WEAPON_CLASSES
                )
                last_frame = frame.copy()
                
                # Alertas en consola
                current_time = datetime.now()
                
                # Alerta de conteo de personas (cada 30 frames)
                if frame_number % 30 == 0:
                    print(f"[{current_time.strftime('%H:%M:%S')}] Personas detectadas: {num_people}")
                    if objects_detected:
                        print(f"[{current_time.strftime('%H:%M:%S')}] Objetos detectados: {len(objects_detected)}")
                
                # Alerta de armas detectadas (evitar spam, máximo cada 2 segundos)
                for weapon_bbox, weapon_label, weapon_conf in weapons_detected:
                    weapon_key = f"{weapon_label}_{weapon_conf:.2f}"
                    if weapon_key not in last_weapon_alert_time:
                        last_weapon_alert_time[weapon_key] = current_time
                        print(f"[{current_time.strftime('%H:%M:%S')}] ALERTA: Arma detectada - {weapon_label.upper()} (Confianza: {weapon_conf:.2f})")
                    else:
                        time_diff = (current_time - last_weapon_alert_time[weapon_key]).total_seconds()
                        if time_diff > 2.0:  # Alerta cada 2 segundos máximo
                            last_weapon_alert_time[weapon_key] = current_time
                            print(f"[{current_time.strftime('%H:%M:%S')}] ALERTA: Arma detectada - {weapon_label.upper()} (Confianza: {weapon_conf:.2f})")
            else:
                if last_frame is not None:
                    frame = last_frame
            
            cv2.imshow("Smart Surveillance - Deteccion de Armas", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        pass
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nSistema cerrado")

if __name__ == "__main__":
    main()
