"""
Script para preparar y organizar el dataset para entrenamiento
Convierte imágenes etiquetadas al formato YOLO
"""

import os
import shutil
from pathlib import Path
import json

def create_dataset_structure():
    """Crea la estructura de directorios para el dataset"""
    base_dir = Path("dataset")
    
    directories = [
        base_dir / "train" / "images",
        base_dir / "train" / "labels",
        base_dir / "val" / "images",
        base_dir / "val" / "labels",
        base_dir / "test" / "images",
        base_dir / "test" / "labels",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Creado: {directory}")
    
    return base_dir


def create_data_yaml():
    """Crea el archivo data.yaml necesario para entrenamiento"""
    yaml_content = """# Configuración del dataset para entrenamiento YOLOv8

path: ./dataset
train: train/images
val: val/images

# Nombres de las clases
# IMPORTANTE: Asegúrate de que las clases coincidan con tus etiquetas
names:
  0: knife   # Cuchillo
  1: gun     # Pistola/Arma de fuego
  2: weapon  # Arma genérica
"""
    
    yaml_path = Path("dataset/data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    print(f"✅ Archivo data.yaml creado: {yaml_path}")
    return yaml_path


def convert_labelme_to_yolo(labelme_json_path, output_label_path, image_width, image_height):
    """
    Convierte anotaciones de LabelMe al formato YOLO
    Formato YOLO: class_id center_x center_y width height (normalizado 0-1)
    """
    with open(labelme_json_path, 'r') as f:
        data = json.load(f)
    
    yolo_lines = []
    
    for shape in data['shapes']:
        label = shape['label'].lower()
        
        # Mapeo de etiquetas a clases
        class_mapping = {
            'knife': 0,
            'cuchillo': 0,
            'gun': 1,
            'pistola': 1,
            'arma': 1,
            'weapon': 2,
            'weapon_gen': 2,
        }
        
        class_id = class_mapping.get(label, -1)
        if class_id == -1:
            print(f"⚠️  Etiqueta desconocida: {label}. Saltando...")
            continue
        
        # Convertir coordenadas de LabelMe a YOLO
        points = shape['points']
        
        if shape['shape_type'] == 'rectangle':
            # Rectángulo: [x1, y1], [x2, y2]
            x1, y1 = points[0]
            x2, y2 = points[1]
            
            # Normalizar coordenadas
            center_x = ((x1 + x2) / 2) / image_width
            center_y = ((y1 + y2) / 2) / image_height
            width = abs(x2 - x1) / image_width
            height = abs(y2 - y1) / image_height
            
            yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
    
    # Guardar archivo YOLO
    with open(output_label_path, 'w') as f:
        f.writelines(yolo_lines)
    
    return len(yolo_lines)


def organize_dataset(source_dir="raw_data", split_ratio=(0.7, 0.2, 0.1)):
    """
    Organiza imágenes y etiquetas en train/val/test
    split_ratio: (train, val, test)
    """
    print("\n" + "=" * 60)
    print("ORGANIZANDO DATASET")
    print("=" * 60)
    
    # Crear estructura
    base_dir = create_dataset_structure()
    create_data_yaml()
    
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"\n❌ Error: No se encontró el directorio '{source_dir}'")
        print("\n📋 Estructura esperada de raw_data:")
        print("raw_data/")
        print("  ├── image1.jpg")
        print("  ├── image1.json (anotaciones LabelMe)")
        print("  ├── image2.jpg")
        print("  ├── image2.json")
        print("  └── ...")
        print("\n💡 Usa LabelMe para etiquetar tus imágenes:")
        print("   pip install labelme")
        print("   labelme")
        return
    
    # Encontrar todas las imágenes
    image_files = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png")) + list(source_path.glob("*.jpeg"))
    
    if not image_files:
        print(f"❌ No se encontraron imágenes en '{source_dir}'")
        return
    
    print(f"\n📸 Encontradas {len(image_files)} imágenes")
    
    # Dividir en train/val/test
    import random
    random.shuffle(image_files)
    
    n_total = len(image_files)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    print(f"📊 División:")
    print(f"   Train: {len(train_files)} ({len(train_files)/n_total*100:.1f}%)")
    print(f"   Val:   {len(val_files)} ({len(val_files)/n_total*100:.1f}%)")
    print(f"   Test:  {len(test_files)} ({len(test_files)/n_total*100:.1f}%)")
    
    # Copiar archivos
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    for split_name, files in splits.items():
        print(f"\n📦 Procesando {split_name}...")
        count = 0
        
        for img_file in files:
            # Buscar archivo JSON correspondiente
            json_file = img_file.with_suffix('.json')
            
            if not json_file.exists():
                print(f"⚠️  No se encontró anotación para {img_file.name}. Saltando...")
                continue
            
            # Copiar imagen
            dest_img = base_dir / split_name / "images" / img_file.name
            shutil.copy2(img_file, dest_img)
            
            # Leer dimensiones de la imagen usando PIL o cv2
            try:
                from PIL import Image
                img = Image.open(img_file)
                img_width, img_height = img.size
            except:
                import cv2
                img = cv2.imread(str(img_file))
                img_height, img_width = img.shape[:2]
            
            # Convertir anotación y guardar
            label_file = base_dir / split_name / "labels" / (img_file.stem + '.txt')
            annotations_count = convert_labelme_to_yolo(json_file, label_file, img_width, img_height)
            
            if annotations_count > 0:
                count += 1
        
        print(f"   ✅ {count} imágenes procesadas para {split_name}")
    
    print("\n" + "=" * 60)
    print("✅ DATASET ORGANIZADO CORRECTAMENTE")
    print("=" * 60)
    print("\n💡 Próximos pasos:")
    print("   1. Revisa dataset/data.yaml y ajusta las clases si es necesario")
    print("   2. Ejecuta: python train.py")
    print()


def download_sample_dataset_info():
    """Información sobre cómo obtener datasets públicos"""
    print("\n" + "=" * 60)
    print("RECURSOS PARA DATASETS DE ARMAS Y CUCHILLOS")
    print("=" * 60)
    print()
    print("📚 Datasets públicos disponibles:")
    print()
    print("1. Roboflow Universe:")
    print("   https://universe.roboflow.com/")
    print("   Busca: 'knife detection', 'weapon detection', 'gun detection'")
    print()
    print("2. Kaggle:")
    print("   https://www.kaggle.com/datasets")
    print("   Busca: 'weapon detection', 'knife detection'")
    print()
    print("3. Crear tu propio dataset:")
    print("   - Usa LabelMe para etiquetar: pip install labelme")
    print("   - Toma fotos de objetos (cuchillos, armas de juguete, etc.)")
    print("   - Etiqueta con cuidado para mejor precisión")
    print()
    print("⚠️  IMPORTANTE:")
    print("   - Usa solo imágenes legales y éticas")
    print("   - Para pruebas, usa armas de juguete o objetos simulados")
    print("   - Respeta las leyes locales sobre armas")
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--info":
            download_sample_dataset_info()
        elif sys.argv[1] == "--create-structure":
            create_dataset_structure()
            create_data_yaml()
        else:
            organize_dataset(sys.argv[1])
    else:
        organize_dataset()

