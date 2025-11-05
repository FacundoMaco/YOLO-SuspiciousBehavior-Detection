"""
Script de entrenamiento mejorado para detectar cuchillos y armas
Especialmente optimizado para mejorar la detección de cuchillos
"""

from ultralytics import YOLO
import os
import yaml
from pathlib import Path

def check_dataset_structure():
    """Verifica y crea la estructura del dataset si no existe"""
    dataset_path = Path("dataset")
    
    print("📁 Verificando estructura del dataset...")
    
    required_dirs = [
        dataset_path / "train" / "images",
        dataset_path / "train" / "labels",
        dataset_path / "val" / "images",
        dataset_path / "val" / "labels",
    ]
    
    for dir_path in required_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {dir_path}")
    
    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        print(f"\n⚠️  No se encontró data.yaml. Creando uno básico...")
        create_data_yaml()
    
    return dataset_path


def create_data_yaml():
    """Crea el archivo data.yaml para entrenamiento de cuchillos"""
    yaml_content = {
        'path': './dataset',
        'train': 'train/images',
        'val': 'val/images',
        'names': {
            0: 'knife',  # Cuchillo - clase principal
            1: 'gun',    # Pistola (opcional)
            2: 'weapon'  # Arma genérica (opcional)
        }
    }
    
    yaml_path = Path("dataset/data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Archivo data.yaml creado: {yaml_path}")
    return yaml_path


def count_annotations(dataset_path):
    """Cuenta las anotaciones disponibles en el dataset"""
    train_labels = list((dataset_path / "train" / "labels").glob("*.txt"))
    val_labels = list((dataset_path / "val" / "labels").glob("*.txt"))
    
    train_images = list((dataset_path / "train" / "images").glob("*.jpg")) + \
                   list((dataset_path / "train" / "images").glob("*.png"))
    val_images = list((dataset_path / "val" / "images").glob("*.jpg")) + \
                 list((dataset_path / "val" / "images").glob("*.png"))
    
    print(f"\n📊 Estadísticas del dataset:")
    print(f"   Train: {len(train_images)} imágenes, {len(train_labels)} etiquetas")
    print(f"   Val:   {len(val_images)} imágenes, {len(val_labels)} etiquetas")
    
    if len(train_labels) == 0:
        print("\n⚠️  ADVERTENCIA: No se encontraron etiquetas de entrenamiento!")
        print("   Necesitas al menos 50-100 imágenes etiquetadas para entrenar.")
        print("   Ejecuta 'prepare_dataset.py' primero o descarga un dataset.")
        return False
    
    if len(train_labels) < 50:
        print(f"\n⚠️  ADVERTENCIA: Solo {len(train_labels)} etiquetas encontradas.")
        print("   Se recomienda al menos 100-200 imágenes para buen rendimiento.")
        print("   El modelo puede funcionar pero con menor precisión.")
    
    return True


def train_model():
    """
    Entrena un modelo YOLOv8 personalizado específicamente para detectar cuchillos
    """
    
    print("=" * 70)
    print("ENTRENAMIENTO DE MODELO PERSONALIZADO PARA DETECCIÓN DE CUCHILLOS")
    print("=" * 70)
    print()
    
    # Verificar estructura del dataset
    dataset_path = check_dataset_structure()
    data_yaml = dataset_path / "data.yaml"
    
    if not data_yaml.exists():
        print("❌ Error: No se pudo crear data.yaml")
        return
    
    # Contar anotaciones
    if not count_annotations(dataset_path):
        response = input("\n¿Deseas continuar de todas formas? (s/n): ").lower()
        if response != 's':
            print("Entrenamiento cancelado.")
            return
    
    print("\n🔄 Cargando modelo base YOLOv8n...")
    try:
        model = YOLO("yolov8n.pt")  # Modelo nano para entrenamiento rápido
        print("✅ Modelo base cargado")
    except Exception as e:
        print(f"❌ Error al cargar modelo base: {e}")
        print("   El modelo se descargará automáticamente...")
        model = YOLO("yolov8n.pt")
    
    print("\n⚙️  Configurando parámetros de entrenamiento...")
    print("   Modelo: YOLOv8n (nano - rápido)")
    print("   Clases: knife (cuchillo), gun (pistola), weapon (arma genérica)")
    print("   Épocas: 100 (con early stopping)")
    print()
    
    # Configuración optimizada para detección de cuchillos
    try:
        results = model.train(
            data=str(data_yaml.absolute()),  # Ruta absoluta al archivo YAML
            epochs=100,                       # Número de épocas
            imgsz=640,                        # Tamaño de imagen (estándar YOLO)
            batch=4,                          # Tamaño del batch (reducido para CPU)
            name='knife_detection',          # Nombre del experimento
            patience=25,                      # Early stopping patience
            save=True,                        # Guardar checkpoints
            save_period=10,                   # Guardar cada N épocas
            plots=True,                      # Generar gráficos
            val=True,                        # Validar durante entrenamiento
            device='cpu',                    # Usar CPU (cambia a 0 si tienes GPU)
            workers=4,                       # Número de workers (reducido para CPU)
            project='runs/detect',           # Directorio del proyecto
            exist_ok=True,                   # Sobrescribir si existe
            pretrained=True,                 # Usar pesos pre-entrenados
            optimizer='AdamW',               # Optimizador
            lr0=0.001,                       # Learning rate inicial
            lrf=0.01,                        # Learning rate final
            momentum=0.937,                  # Momentum
            weight_decay=0.0005,            # Weight decay
            warmup_epochs=3,                 # Épocas de warmup
            warmup_momentum=0.8,             # Momentum durante warmup
            warmup_bias_lr=0.1,             # Learning rate de bias durante warmup
            box=7.5,                        # Loss de bounding box
            cls=0.5,                        # Loss de clasificación
            dfl=1.5,                        # Loss de distribución focal
            # Aumentos de datos específicos para mejorar detección de cuchillos
            hsv_h=0.015,                    # Aumento de matiz HSV
            hsv_s=0.7,                      # Aumento de saturación HSV
            hsv_v=0.4,                      # Aumento de valor HSV
            degrees=10.0,                   # Rotación máxima
            translate=0.1,                  # Traslación
            scale=0.5,                      # Escala
            shear=2.0,                      # Cizallamiento
            perspective=0.0,                # Perspectiva
            flipud=0.0,                     # Volteo vertical
            fliplr=0.5,                     # Volteo horizontal
            mosaic=1.0,                     # Mosaic augmentation
            mixup=0.1,                      # Mixup augmentation
            copy_paste=0.1,                 # Copy-paste augmentation
        )
        
        print("\n" + "=" * 70)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("=" * 70)
        print()
        print(f"📊 Resultados guardados en: {results.save_dir}")
        print(f"🎯 Mejor modelo: {results.save_dir}/weights/best.pt")
        print(f"📈 Último checkpoint: {results.save_dir}/weights/last.pt")
        print()
        
        # Evaluar el modelo
        print("📈 Evaluando modelo final...")
        metrics = model.val()
        print(f"   ✅ mAP50: {metrics.box.map50:.4f}")
        print(f"   ✅ mAP50-95: {metrics.box.map:.4f}")
        print()
        
        # Copiar el mejor modelo a la raíz
        import shutil
        best_model = Path(results.save_dir) / "weights" / "best.pt"
        if best_model.exists():
            dest_model = Path("best.pt")
            shutil.copy2(best_model, dest_model)
            print(f"✅ Modelo copiado a: {dest_model.absolute()}")
            print("   El script main.py lo detectará automáticamente.")
            print()
        
        print("💡 Próximos pasos:")
        print("   1. Prueba el modelo con: python main.py")
        print("   2. Si la precisión no es suficiente, añade más imágenes al dataset")
        print("   3. Re-entrena con más épocas o ajusta los parámetros")
        print()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Entrenamiento interrumpido por el usuario")
        print("   Los checkpoints guardados hasta ahora están disponibles en runs/detect/")
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        print("   Verifica que:")
        print("   - El dataset esté correctamente estructurado")
        print("   - Las etiquetas estén en formato YOLO")
        print("   - Tengas suficiente espacio en disco")
        print("   - Tu GPU tenga suficiente memoria (o usa device='cpu')")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--create-yaml":
            create_data_yaml()
        elif sys.argv[1] == "--check":
            check_dataset_structure()
            count_annotations(Path("dataset"))
        else:
            print(f"Uso: python {sys.argv[0]} [--create-yaml|--check]")
    else:
        train_model()


