"""
Script para entrenar el modelo CNN de clasificación de actividades
Basado en keypoints de pose extraídos de videos
"""

import numpy as np
import os
from models.activity_classifier import ActivityClassifier
import pickle


def load_training_data(data_dir='data/activities'):
    """
    Carga datos de entrenamiento desde archivos
    
    Estructura esperada:
    data/activities/
    ├── caminar/
    │   ├── keypoints_001.npy
    │   └── ...
    ├── sentarse/
    ├── interactuar/
    ├── saludar/
    └── hurto/
    """
    X_train = []
    y_train = []
    
    activities = ['caminar', 'sentarse', 'interactuar', 'saludar', 'hurto']
    
    print("📦 Cargando datos de entrenamiento...")
    
    for activity_idx, activity in enumerate(activities):
        activity_dir = os.path.join(data_dir, activity)
        
        if not os.path.exists(activity_dir):
            print(f"⚠️  Directorio no encontrado: {activity_dir}")
            continue
        
        # Cargar archivos .npy con keypoints
        files = [f for f in os.listdir(activity_dir) if f.endswith('.npy')]
        
        print(f"  {activity}: {len(files)} muestras")
        
        for file in files:
            file_path = os.path.join(activity_dir, file)
            try:
                keypoints = np.load(file_path)
                
                # Asegurar que tenga la forma correcta (17, 2) o (34,)
                if keypoints.shape == (17, 2):
                    keypoints = keypoints.flatten()
                elif keypoints.shape == (17, 3):
                    keypoints = keypoints[:, :2].flatten()
                
                # Normalizar
                if np.max(np.abs(keypoints)) > 0:
                    keypoints = keypoints / (np.max(np.abs(keypoints)) + 1e-8)
                
                X_train.append(keypoints)
                
                # One-hot encoding
                y_one_hot = np.zeros(len(activities))
                y_one_hot[activity_idx] = 1.0
                y_train.append(y_one_hot)
                
            except Exception as e:
                print(f"    Error cargando {file}: {e}")
                continue
    
    if len(X_train) == 0:
        print("❌ No se encontraron datos de entrenamiento")
        return None, None
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"\n✅ Datos cargados: {len(X_train)} muestras")
    print(f"   Forma X: {X_train.shape}")
    print(f"   Forma y: {y_train.shape}")
    
    return X_train, y_train


def train_activity_model(data_dir='data/activities', epochs=20, batch_size=32):
    """
    Entrena el modelo de clasificación de actividades
    
    Args:
        data_dir: Directorio con los datos de entrenamiento
        epochs: Número de épocas
        batch_size: Tamaño del batch
    """
    print("=" * 70)
    print("ENTRENAMIENTO DEL MODELO DE CLASIFICACIÓN DE ACTIVIDADES")
    print("=" * 70)
    print()
    
    # Cargar datos
    X_train, y_train = load_training_data(data_dir)
    
    if X_train is None:
        print("\n❌ No se pueden cargar los datos. Asegúrate de tener la estructura correcta.")
        print("\nEstructura esperada:")
        print("data/activities/")
        print("├── caminar/")
        print("│   ├── keypoints_001.npy")
        print("│   └── ...")
        print("├── sentarse/")
        print("├── interactuar/")
        print("├── saludar/")
        print("└── hurto/")
        return
    
    # Dividir en train/val (80/20)
    split_idx = int(len(X_train) * 0.8)
    indices = np.random.permutation(len(X_train))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    X_train_split = X_train[train_indices]
    y_train_split = y_train[train_indices]
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]
    
    print(f"\n📊 División de datos:")
    print(f"   Entrenamiento: {len(X_train_split)} muestras")
    print(f"   Validación: {len(X_val)} muestras")
    
    # Crear y entrenar modelo
    print("\n🔄 Creando modelo...")
    classifier = ActivityClassifier()
    
    print("\n🚀 Iniciando entrenamiento...")
    print(f"   Épocas: {epochs}")
    print(f"   Batch size: {batch_size}")
    print()
    
    history = classifier.train(
        X_train_split, y_train_split,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Guardar modelo
    model_path = 'models/activity_model.h5'
    os.makedirs('models', exist_ok=True)
    classifier.save_model(model_path)
    
    print("\n" + "=" * 70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    print(f"📁 Modelo guardado en: {model_path}")
    print("\n💡 El modelo se cargará automáticamente en main.py")


if __name__ == "__main__":
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/activities'
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    
    train_activity_model(data_dir, epochs, batch_size)

