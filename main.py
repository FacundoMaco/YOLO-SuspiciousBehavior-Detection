import cv2
import numpy as np
from ultralytics import YOLO

# Cargar modelos
print("Cargando modelos YOLOv8...")
try:
    model_objects = YOLO("yolov8n.pt")
    print("✅ Modelo de objetos cargado")
except Exception as e:
    print(f"❌ Error al cargar modelo de objetos: {e}")
    exit(1)

try:
    model_pose = YOLO("yolov8n-pose.pt")
    print("✅ Modelo de pose cargado")
except Exception as e:
    print(f"❌ Error al cargar modelo de pose: {e}")
    model_pose = YOLO("yolov8n-pose.pt")

# Intentar cargar modelo personalizado entrenado para cuchillos
model_custom = None
try:
    model_custom = YOLO("best.pt")
    print("✅ Modelo personalizado cargado (para detección de cuchillos)")
except:
    print("⚠️  Modelo personalizado no encontrado (best.pt)")
    print("   Ejecuta train.py para entrenar un modelo con dataset de cuchillos")

print("✅ Todos los modelos cargados correctamente\n")


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
    
    # Calcular intersección
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calcular áreas
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
    center2_x = (x1_2 + x2_2) / 2
    center2_y = (y1_2 + y2_2) / 2
    
    distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    return distance


def detect_objects_and_pose(frame, model_objects, model_pose, model_custom=None):
    """Función básica de detección con soporte para modelo personalizado y detección de personas sospechosas"""
    detected_threats = []
    alerts = []
    knives_detected = []  # Lista de cuchillos detectados [(bbox, label, conf)]
    
    try:
        # Primero usar modelo personalizado si está disponible (más preciso para cuchillos)
        if model_custom is not None:
            results_custom = model_custom(frame, verbose=False, conf=0.3)
            for r in results_custom:
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        label = model_custom.names[class_id]
                        
                        if confidence > 0.3:  # Umbral más bajo para mejor detección
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            bbox = (x1, y1, x2, y2)
                            color = (0, 0, 255)  # Rojo para amenazas
                            
                            # Guardar cuchillo detectado
                            if 'knife' in label.lower() or 'cuchillo' in label.lower():
                                knives_detected.append((bbox, label, confidence))
                            
                            detected_threats.append(label)
                            alerts.append(f"AMENAZA_DETECTADA: {label}")
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                            cv2.putText(frame, f"{label} {confidence:.2f}", 
                                      (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 3)
        
        # Detección de objetos estándar (personas y otros)
        results_objects = model_objects(frame, verbose=False, conf=0.5)
        
        # Detección de pose
        results_pose = model_pose(frame, verbose=False, conf=0.5)
        
        # Procesar detección de objetos estándar
        for r in results_objects:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    label = model_objects.names[class_id]
                    
                    if confidence > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bbox = (x1, y1, x2, y2)
                        
                        # Detectar objetos peligrosos del modelo estándar
                        if label.lower() in ['knife', 'gun', 'scissors', 'baseball bat']:
                            color = (0, 0, 255)  # Rojo para amenazas
                            detected_threats.append(label)
                            alerts.append(f"AMENAZA_DETECTADA: {label}")
                            
                            # Guardar cuchillo detectado
                            if label.lower() == 'knife':
                                knives_detected.append((bbox, label, confidence))
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, f"{label} {confidence:.2f}", 
                                      (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        elif label == 'person':
                            # Las personas se procesarán con el modelo de pose
                            pass
                        else:
                            color = (255, 255, 0)  # Amarillo para otros
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, f"{label} {confidence:.2f}", 
                                      (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Procesar detección de pose y asociar con personas
        suspicious_persons = set()  # Índices de personas sospechosas
        
        for r in results_pose:
            if r.keypoints is not None and r.boxes is not None and len(r.boxes) > 0:
                for i in range(min(len(r.boxes), len(r.keypoints.data))):
                    try:
                        # Obtener bbox de la persona
                        box = r.boxes[i]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bbox = (x1, y1, x2, y2)
                        
                        # Obtener keypoints
                        kpts = r.keypoints.data[i]
                        if hasattr(kpts, 'cpu'):
                            keypoints_data = kpts.cpu().numpy()
                        else:
                            keypoints_data = np.array(kpts)
                        
                        # Extraer coordenadas (x, y) y confianza
                        if keypoints_data.shape[1] >= 3:
                            keypoints = keypoints_data[:, :2]  # Solo x, y
                            confidences = keypoints_data[:, 2]  # Confianzas
                        elif keypoints_data.shape[1] >= 2:
                            keypoints = keypoints_data[:, :2]
                            confidences = np.ones(len(keypoints))
                        else:
                            continue
                        
                        # Filtrar keypoints con baja confianza
                        valid_mask = confidences > 0.25
                        valid_count = np.sum(valid_mask)
                        
                        if valid_count >= 5:  # Al menos 5 puntos válidos
                            # Marcar inválidos como (0, 0)
                            keypoints[~valid_mask] = [0, 0]
                            
                            # Verificar si esta persona tiene un cuchillo cerca
                            is_suspicious = False
                            for knife_bbox, knife_label, knife_conf in knives_detected:
                                # Calcular IoU y distancia
                                iou = calculate_iou(bbox, knife_bbox)
                                distance = calculate_distance(bbox, knife_bbox)
                                
                                # Calcular tamaño promedio de la persona para umbral dinámico
                                person_size = np.sqrt((x2 - x1) * (y2 - y1))
                                
                                # Una persona es sospechosa si:
                                # - El cuchillo está dentro de su bbox (IoU > 0) O
                                # - El cuchillo está muy cerca (distancia < 30% del tamaño de la persona)
                                if iou > 0 or distance < person_size * 0.3:
                                    is_suspicious = True
                                    suspicious_persons.add(i)
                                    
                                    # Agregar alerta
                                    if f"PERSONA_SOSPECHOSA: Persona con {knife_label}" not in alerts:
                                        alerts.append(f"PERSONA_SOSPECHOSA: Persona con {knife_label}")
                                        detected_threats.append(f"Persona con {knife_label}")
                                    break
                            
                            # Elegir color según si es sospechosa
                            if is_suspicious:
                                # Rojo/Naranja para personas sospechosas
                                person_color = (0, 0, 255)  # Rojo
                                skeleton_color = (255, 0, 0)  # Azul para el esqueleto (contraste)
                            else:
                                # Verde para personas normales
                                person_color = (0, 255, 0)  # Verde
                                skeleton_color = (255, 0, 0)  # Azul para el esqueleto
                            
                            # Dibujar esqueleto
                            frame = draw_pose_skeleton(frame, bbox, keypoints)
                            
                            # Dibujar bbox de la persona
                            thickness = 3 if is_suspicious else 2
                            cv2.rectangle(frame, (x1, y1), (x2, y2), person_color, thickness)
                            
                            # Etiqueta especial para personas sospechosas
                            if is_suspicious:
                                cv2.putText(frame, "SOSPECHOSO", (x1, y1 - 30), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                    
                    except Exception as e:
                        continue
        
    except Exception as e:
        print(f"Error en detección: {e}")
    
    return frame, detected_threats, alerts


def main():
    print("Sistema de Detección de Pose y Comportamiento Sospechoso")
    print("Características:")
    print("  - Detección de pose humana")
    print("  - Detección de cuchillos y armas")
    print("  - Identificación de personas sospechosas (personas con cuchillos)")
    print("\nOpciones:")
    print("  - Presiona Enter para usar cámara web")
    print("  - O ingresa la ruta a un archivo de video")
    
    source = input("\nFuente de video [Enter para cámara]: ").strip()
    
    if source == "":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la fuente de video")
        return
    
    print("\n✅ Sistema activo. Presiona 'q' para salir")
    print("⚠️  Las personas con cuchillos serán marcadas como SOSPECHOSAS\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del video o error al leer frame")
            break
        
        # Detectar objetos y pose
        frame, threats, alerts = detect_objects_and_pose(frame, model_objects, model_pose, model_custom)
        
        # Mostrar alertas en pantalla
        y_offset = 30
        for alert in alerts[-3:]:  # Mostrar últimas 3 alertas
            cv2.putText(frame, alert, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30
        
        # Imprimir alertas críticas en consola (solo alertas nuevas)
        suspicious_alerts = [a for a in alerts if "PERSONA_SOSPECHOSA" in a]
        if suspicious_alerts:
            print(f"🚨 {suspicious_alerts[-1]}")
        elif threats:
            threat_alerts = [t for t in threats if "Persona con" not in t]
            if threat_alerts:
                print(f"⚠️ AMENAZAS DETECTADAS: {', '.join(set(threat_alerts))}")
        
        cv2.imshow("AI Vision - Detección de Pose", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Sistema cerrado correctamente")


if __name__ == "__main__":
    main()
