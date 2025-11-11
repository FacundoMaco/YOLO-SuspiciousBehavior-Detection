"""
Sistema de Videovigilancia Inteligente para Detección de Actividades Anómalas
Basado en el artículo de Sathiyavathi et al. (2021)
Integra CNN, OpenPose (YOLOv8-pose), clasificación de riesgo y API REST
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import os

# Importar módulos del sistema
from models.activity_classifier import ActivityClassifier
from utils.risk_classifier import RiskClassifier
from utils.temporal_analyzer import TemporalAnalyzer
from database.event_db import EventDatabase
from api.alert_api import AlertAPI

# ============================================================================
# CONFIGURACIÓN Y CARGA DE MODELOS
# ============================================================================

print("=" * 70)
print("SISTEMA DE VIDEOVIGILANCIA INTELIGENTE")
print("Prevención de Robos, Asaltos y Ataques")
print("Detección de Actividades Anómalas usando CNN")
print("=" * 70)
print()

# Cargar modelos YOLOv8
print("📦 Cargando modelos YOLOv8...")

# Modelo de pose para detectar personas
try:
    model_pose = YOLO("yolov8n-pose.pt")
    print("✅ Modelo de pose cargado")
except Exception as e:
    print(f"❌ Error al cargar modelo de pose: {e}")
    model_pose = YOLO("yolov8n-pose.pt")

# Modelo personalizado para detectar armas (cuchillos, pistolas, etc.)
model_weapons = None
try:
    model_weapons = YOLO("best.pt")
    print("✅ Modelo de detección de armas cargado")
except:
    print("⚠️  Modelo personalizado de armas no encontrado (best.pt)")
    print("   El sistema funcionará solo con detección de pose y actividades")
    print("   Para mejor detección de armas, ejecuta train.py para entrenar un modelo")

# Cargar clasificador de actividades CNN
print("\n📦 Cargando clasificador de actividades CNN...")
try:
    activity_classifier = ActivityClassifier(model_path='models/activity_model.h5')
    print("✅ Clasificador de actividades CNN cargado (modelo entrenado)")
except Exception as e:
    print(f"⚠️  Modelo de actividades no encontrado")
    print(f"   Usando clasificación basada en reglas (sin entrenamiento necesario)")
    activity_classifier = ActivityClassifier()

# Inicializar componentes del sistema
print("\n📦 Inicializando componentes del sistema...")
risk_classifier = RiskClassifier()
temporal_analyzer = TemporalAnalyzer(sequence_length=30)
event_database = EventDatabase()

# Configurar API REST con opción de enviar a app externa (Lovable/Supabase)
try:
    from config_lovable import LOVABLE_API_URL, LOVABLE_API_KEY, CV_API_KEY
    print(f"📋 Configuración cargada desde config_lovable.py")
except ImportError:
    # Si no existe config_lovable.py, usar variables de entorno
    LOVABLE_API_URL = os.getenv('LOVABLE_API_URL', None)
    LOVABLE_API_KEY = os.getenv('LOVABLE_API_KEY', None)
    CV_API_KEY = None

# Mostrar información de configuración
if LOVABLE_API_URL and LOVABLE_API_URL != 'https://TU_PROYECTO.supabase.co/functions/v1/receive-cv-alert':
    print(f"📡 Configurado para enviar alertas a: {LOVABLE_API_URL}")
    if LOVABLE_API_KEY:
        print(f"🔑 API Key configurado: {'*' * 20}...{LOVABLE_API_KEY[-8:]}")
else:
    print("⚠️  LOVABLE_API_URL no configurada")
    print("   Edita config_lovable.py y configura tu URL de Supabase")
    print("   O exporta: export LOVABLE_API_URL='https://tu-proyecto.supabase.co/functions/v1/receive-cv-alert'")

alert_api = AlertAPI(
    host='localhost', 
    port=5000,
    external_api_url=LOVABLE_API_URL if LOVABLE_API_URL and 'TU_PROYECTO' not in LOVABLE_API_URL else None,
    external_api_key=LOVABLE_API_KEY
)

# Iniciar API REST en thread separado (si Flask está disponible)
print("\n🌐 Iniciando API REST...")
try:
    alert_api.run(threaded=True)
except Exception as e:
    print(f"⚠️  API REST no disponible: {e}")
    print("   El sistema funcionará sin API REST")

print("\n✅ Todos los componentes cargados correctamente\n")

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def draw_pose_skeleton(frame, bbox, keypoints):
    """Dibuja el esqueleto de la persona con sus articulaciones"""
    if bbox is None or len(keypoints) == 0:
        return frame
    
    try:
        x1, y1, x2, y2 = bbox
        frame_h, frame_w = frame.shape[:2]
        
        # Validar bbox
        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > frame_w or y2 > frame_h:
            return frame
        
        # Conexiones del esqueleto (COCO pose format - 17 puntos clave)
        skeleton_connections = [
            [0, 1], [0, 2], [1, 3], [2, 4],  # Cabeza
            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # Brazos
            [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],  # Piernas
            [5, 11], [6, 12]  # Torso
        ]
        
        # Dibujar conexiones primero
        for connection in skeleton_connections:
            if connection[0] < len(keypoints) and connection[1] < len(keypoints):
                pt1 = keypoints[connection[0]]
                pt2 = keypoints[connection[1]]
                
                # Validar que ambos puntos sean válidos (no 0,0)
                if (pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0):
                    pt1_int = (int(pt1[0]), int(pt1[1]))
                    pt2_int = (int(pt2[0]), int(pt2[1]))
                    
                    # Validar que estén dentro del frame
                    if (0 <= pt1_int[0] < frame_w and 0 <= pt1_int[1] < frame_h and
                        0 <= pt2_int[0] < frame_w and 0 <= pt2_int[1] < frame_h):
                        cv2.line(frame, pt1_int, pt2_int, (255, 0, 0), 2)
        
        # Dibujar puntos clave
        for kpt in keypoints:
            if kpt[0] > 0 and kpt[1] > 0:
                x, y = int(kpt[0]), int(kpt[1])
                if 0 <= x < frame_w and 0 <= y < frame_h:
                    cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
        
        # Dibujar centro de la persona (eje)
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        if 0 <= center_x < frame_w and 0 <= center_y < frame_h:
            cv2.circle(frame, (center_x, center_y), 8, (0, 0, 255), -1)
    
    except Exception as e:
        pass
    
    return frame


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


def calculate_distance(box1, box2):
    """Calcula la distancia entre los centros de dos bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    center1_x = (x1_1 + x2_1) / 2
    center1_y = (y1_1 + y2_1) / 2
    center2_x = (x2_2 + x1_2) / 2
    center2_y = (y2_2 + y1_2) / 2
    
    distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    return distance


# ============================================================================
# FUNCIÓN PRINCIPAL DE DETECCIÓN
# ============================================================================

def detect_and_classify(frame, model_pose, model_weapons=None,
                       activity_classifier=None, risk_classifier=None,
                       temporal_analyzer=None, event_database=None, alert_api=None,
                       frame_number=0, location="Desconocida"):
    """
    Detecta personas, armas, clasifica actividades y genera alertas
    Enfocado en prevenir robos, asaltos y ataques
    
    Returns:
        frame: Frame procesado con anotaciones
        detected_events: Lista de eventos detectados
    """
    detected_events = []
    weapons_detected = []  # Lista de armas detectadas [(bbox, label, conf)]
    person_data = {}  # {person_id: {keypoints, bbox, has_weapon}}
    
    try:
        # 1. Detección de armas (cuchillos, pistolas, etc.)
        if model_weapons is not None:
            results_weapons = model_weapons(frame, verbose=False, conf=0.3)
            for r in results_weapons:
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        label = model_weapons.names[class_id]
                        
                        if confidence > 0.3:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            bbox = (x1, y1, x2, y2)
                            
                            # Cualquier arma detectada
                            weapons_detected.append((bbox, label, confidence))
                            
                            # Dibujar arma detectada
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(frame, f"ARMA: {label} {confidence:.2f}", 
                                      (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
        
        # 2. Detección de pose y clasificación de actividades
        results_pose = model_pose(frame, verbose=False, conf=0.5)
        
        for r in results_pose:
            if r.keypoints is not None and r.boxes is not None and len(r.boxes) > 0:
                for i in range(min(len(r.boxes), len(r.keypoints.data))):
                    try:
                        # Obtener bbox y keypoints
                        box = r.boxes[i]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bbox = (x1, y1, x2, y2)
                        person_id = i
                        
                        # Obtener keypoints
                        kpts = r.keypoints.data[i]
                        if hasattr(kpts, 'cpu'):
                            keypoints_data = kpts.cpu().numpy()
                        else:
                            keypoints_data = np.array(kpts)
                        
                        # Extraer coordenadas (x, y)
                        if keypoints_data.shape[1] >= 3:
                            keypoints = keypoints_data[:, :2]
                            confidences = keypoints_data[:, 2]
                        elif keypoints_data.shape[1] >= 2:
                            keypoints = keypoints_data[:, :2]
                            confidences = np.ones(len(keypoints))
                        else:
                            continue
                        
                        # Filtrar keypoints válidos
                        valid_mask = confidences > 0.25
                        valid_count = np.sum(valid_mask)
                        
                        if valid_count >= 5:
                            keypoints[~valid_mask] = [0, 0]
                            
                            # Verificar si tiene arma cerca
                            has_weapon = False
                            for weapon_bbox, weapon_label, weapon_conf in weapons_detected:
                                iou = calculate_iou(bbox, weapon_bbox)
                                distance = calculate_distance(bbox, weapon_bbox)
                                person_size = np.sqrt((x2 - x1) * (y2 - y1))
                                
                                # Arma está cerca si está dentro del bbox o muy cerca
                                if iou > 0 or distance < person_size * 0.3:
                                    has_weapon = True
                                    break
                            
                            # Clasificar actividad usando CNN
                            activity_result = None
                            if activity_classifier:
                                activity_result = activity_classifier.predict(keypoints)
                                activity = activity_result['activity']
                                activity_confidence = activity_result['confidence']
                            else:
                                # Fallback: clasificación simple basada en pose
                                activity = 'caminar'  # Por defecto
                                activity_confidence = 0.5
                            
                            # Actualizar análisis temporal
                            if temporal_analyzer:
                                temporal_analyzer.update_sequence(person_id, activity, activity_confidence)
                                temporal_analysis = temporal_analyzer.analyze_sequence(person_id)
                            else:
                                temporal_analysis = None
                            
                            # Clasificar nivel de riesgo
                            if risk_classifier:
                                risk_result = risk_classifier.classify_risk(
                                    activity, has_weapon, activity_confidence
                                )
                                risk_level = risk_result['risk_level']
                                risk_color = risk_classifier.get_risk_color(risk_level)
                                risk_label = risk_classifier.get_risk_label(risk_level)
                            else:
                                risk_level = 'segura' if not has_weapon else 'delictiva'
                                risk_color = (0, 255, 0) if not has_weapon else (0, 0, 255)
                                risk_label = 'SEGURA' if not has_weapon else 'DELICTIVA'
                            
                            # Guardar datos de la persona
                            person_data[person_id] = {
                                'keypoints': keypoints.tolist(),
                                'bbox': bbox,
                                'has_weapon': has_weapon,
                                'activity': activity,
                                'activity_confidence': activity_confidence,
                                'risk_level': risk_level
                            }
                            
                            # Dibujar esqueleto
                            frame = draw_pose_skeleton(frame, bbox, keypoints)
                            
                            # Dibujar bbox con color según riesgo
                            thickness = 3 if risk_level == 'delictiva' else 2
                            cv2.rectangle(frame, (x1, y1), (x2, y2), risk_color, thickness)
                            
                            # Mostrar información
                            info_text = f"{risk_label} | {activity.upper()}"
                            cv2.putText(frame, info_text, (x1, y1 - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, risk_color, 2)
                            
                            # Mostrar número de personas
                            cv2.putText(frame, f"Persona {person_id + 1}", (x1, y2 + 20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
                            # Crear evento si es anómala o delictiva
                            if risk_level in ['anómala', 'delictiva']:
                                event = {
                                    'person_id': person_id,
                                    'activity': activity,
                                    'risk_level': risk_level,
                                    'confidence': activity_confidence,
                                    'has_weapon': has_weapon,
                                    'location': location,
                                    'timestamp': datetime.now().isoformat(),
                                    'keypoints': keypoints.tolist()
                                }
                                detected_events.append(event)
                                
                                # Guardar en base de datos
                                if event_database:
                                    event_id = event_database.insert_event(
                                        activity=activity,
                                        risk_level=risk_level,
                                        confidence=activity_confidence,
                                        person_id=person_id,
                                        location=location,
                                        has_weapon=has_weapon,
                                        keypoints=keypoints.tolist()
                                    )
                                    
                                    # Enviar alerta si es delictiva o anómala con alta confianza
                                    if risk_level == 'delictiva' or (risk_level == 'anómala' and activity_confidence > 0.7):
                                        if alert_api:
                                            alert_api.send_alert(
                                                activity=activity,
                                                risk_level=risk_level,
                                                confidence=activity_confidence,
                                                person_id=person_id,
                                                location=location,
                                                has_weapon=has_weapon,
                                                keypoints=keypoints.tolist()
                                            )
                                            event_database.mark_alert_sent(event_id)
                    
                    except Exception as e:
                        continue
        
        # Mostrar estadísticas en pantalla
        num_people = len(person_data)
        if num_people > 0:
            cv2.putText(frame, f"Personas detectadas: {num_people}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if detected_events:
            cv2.putText(frame, f"Eventos detectados: {len(detected_events)}", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    except Exception as e:
        print(f"Error en detección: {e}")
    
    return frame, detected_events


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    print("=" * 70)
    print("SISTEMA DE DETECCIÓN DE ACTIVIDADES ANÓMALAS")
    print("Prevención de Robos, Asaltos y Ataques")
    print("Basado en CNN y OpenPose (YOLOv8-pose)")
    print("=" * 70)
    print("\nCaracterísticas:")
    print("  ✅ Detección de pose humana")
    print("  ✅ Detección de armas (cuchillos, pistolas, etc.)")
    print("  ✅ Clasificación de actividades (CNN)")
    print("  ✅ Clasificación de riesgo (segura/anómala/delictiva)")
    print("  ✅ Análisis temporal de secuencias")
    print("  ✅ Base de datos de eventos")
    print("  ✅ API REST para alertas")
    print("\nOpciones:")
    print("  - Presiona Enter para usar cámara web")
    print("  - O ingresa la ruta a un archivo de video")
    
    source = input("\nFuente de video [Enter para cámara]: ").strip()
    location = input("Ubicación [Enter para 'Desconocida']: ").strip() or "Desconocida"
    
    if source == "":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("❌ Error: No se pudo abrir la fuente de video")
        return
    
    print("\n✅ Sistema activo. Presiona 'q' para salir")
    print("📊 Estadísticas disponibles en: http://localhost:5000/stats")
    print("📡 API REST disponible en: http://localhost:5000\n")
    
    frame_number = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Fin del video o error al leer frame")
                break
            
            frame_number += 1
            
            # Detectar y clasificar
            frame, events = detect_and_classify(
                frame, model_pose, model_weapons,
                activity_classifier, risk_classifier, temporal_analyzer,
                event_database, alert_api, frame_number, location
            )
            
            # Mostrar eventos críticos en consola
            for event in events:
                if event['risk_level'] == 'delictiva':
                    print(f"🚨 [{event['timestamp']}] ALERTA DELICTIVA: {event['activity']} "
                          f"(Confianza: {event['confidence']:.2f}, Persona: {event['person_id']})")
                elif event['risk_level'] == 'anómala':
                    print(f"⚠️  [{event['timestamp']}] ACTIVIDAD ANÓMALA: {event['activity']} "
                          f"(Confianza: {event['confidence']:.2f}, Persona: {event['person_id']})")
            
            # Mostrar frame
            cv2.imshow("Smart Surveillance - Detección de Actividades Anómalas", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Sistema interrumpido por el usuario")
    
    finally:
        # Cerrar recursos
        cap.release()
        cv2.destroyAllWindows()
        
        # Mostrar estadísticas finales
        if event_database:
            stats = event_database.get_statistics(days=1)
            print("\n" + "=" * 70)
            print("ESTADÍSTICAS DEL SISTEMA")
            print("=" * 70)
            print(f"Total de eventos: {stats['total_events']}")
            print(f"  - Seguros: {stats['safe_events']}")
            print(f"  - Anómalos: {stats['anomalous_events']}")
            print(f"  - Delictivos: {stats['criminal_events']}")
            print(f"Confianza promedio: {stats['avg_confidence']:.2f}")
            print(f"Alertas enviadas: {stats['alerts_sent']}")
            print("=" * 70)
        
        event_database.close()
        print("\n✅ Sistema cerrado correctamente")


if __name__ == "__main__":
    main()
