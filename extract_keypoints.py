"""
Script de utilidad para extraer keypoints de videos
Útil para preparar datos de entrenamiento del modelo de actividades
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import argparse


def extract_keypoints_from_video(video_path, output_dir, activity_name, max_frames=None):
    """
    Extrae keypoints de un video y los guarda como archivos .npy
    
    Args:
        video_path: Ruta al video
        output_dir: Directorio donde guardar los keypoints
        activity_name: Nombre de la actividad (caminar, sentarse, etc.)
        max_frames: Número máximo de frames a procesar (None = todos)
    """
    # Cargar modelo de pose
    print(f"📦 Cargando modelo YOLOv8-pose...")
    model_pose = YOLO("yolov8n-pose.pt")
    
    # Crear directorio de salida
    activity_dir = os.path.join(output_dir, activity_name)
    os.makedirs(activity_dir, exist_ok=True)
    
    # Abrir video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir el video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\n📹 Video: {video_path}")
    print(f"   Frames totales: {total_frames}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Actividad: {activity_name}")
    print(f"   Directorio de salida: {activity_dir}\n")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if max_frames and frame_count >= max_frames:
            break
        
        # Detectar pose
        results = model_pose(frame, verbose=False, conf=0.5)
        
        # Procesar cada persona detectada
        for r in results:
            if r.keypoints is not None and r.boxes is not None and len(r.boxes) > 0:
                for i in range(min(len(r.boxes), len(r.keypoints.data))):
                    try:
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
                        
                        # Filtrar keypoints válidos (al menos 5 puntos con confianza > 0.25)
                        valid_mask = confidences > 0.25
                        valid_count = np.sum(valid_mask)
                        
                        if valid_count >= 5:
                            # Guardar keypoints
                            filename = f"keypoints_{saved_count:06d}.npy"
                            filepath = os.path.join(activity_dir, filename)
                            np.save(filepath, keypoints)
                            saved_count += 1
                            
                            if saved_count % 10 == 0:
                                print(f"  Guardados {saved_count} keypoints...")
                    
                    except Exception as e:
                        continue
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"  Procesados {frame_count}/{total_frames} frames...")
    
    cap.release()
    
    print(f"\n✅ Extracción completada:")
    print(f"   Frames procesados: {frame_count}")
    print(f"   Keypoints guardados: {saved_count}")
    print(f"   Ubicación: {activity_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Extrae keypoints de videos para entrenar el modelo de actividades'
    )
    parser.add_argument('video_path', help='Ruta al video')
    parser.add_argument('activity_name', help='Nombre de la actividad (caminar, sentarse, etc.)')
    parser.add_argument('--output-dir', default='data/activities', 
                       help='Directorio de salida (default: data/activities)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Número máximo de frames a procesar')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("EXTRACCIÓN DE KEYPOINTS PARA ENTRENAMIENTO")
    print("=" * 70)
    print()
    
    extract_keypoints_from_video(
        args.video_path,
        args.output_dir,
        args.activity_name,
        args.max_frames
    )


if __name__ == "__main__":
    main()

