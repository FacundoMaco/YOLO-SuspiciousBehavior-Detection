"""
Script para descargar y preparar datasets de cuchillos para entrenamiento
Proporciona enlaces y guías para obtener datasets públicos
"""

import os
import requests
from pathlib import Path
import json

def print_dataset_sources():
    """Muestra fuentes de datasets públicos de cuchillos"""
    print("=" * 70)
    print("FUENTES DE DATASETS PARA ENTRENAMIENTO DE DETECCIÓN DE CUCHILLOS")
    print("=" * 70)
    print()
    
    sources = [
        {
            "nombre": "Roboflow Universe",
            "descripcion": "Plataforma con múltiples datasets de detección de armas y cuchillos",
            "url": "https://universe.roboflow.com/",
            "buscar": ["knife detection", "weapon detection", "knife dataset"],
            "formato": "YOLO (compatible)",
            "gratis": True
        },
        {
            "nombre": "Kaggle",
            "descripcion": "Plataforma con datasets públicos de machine learning",
            "url": "https://www.kaggle.com/datasets",
            "buscar": ["knife detection", "weapon detection", "security detection"],
            "formato": "Variable (puede necesitar conversión)",
            "gratis": True
        },
        {
            "nombre": "GitHub - Datasets públicos",
            "descripcion": "Repositorios con datasets de seguridad",
            "url": "https://github.com/topics/weapon-detection-dataset",
            "buscar": ["weapon detection", "knife detection", "security dataset"],
            "formato": "Variable",
            "gratis": True
        }
    ]
    
    for i, source in enumerate(sources, 1):
        print(f"{i}. {source['nombre']}")
        print(f"   Descripción: {source['descripcion']}")
        print(f"   URL: {source['url']}")
        print(f"   Buscar: {', '.join(source['buscar'])}")
        print(f"   Formato: {source['formato']}")
        print(f"   Gratis: {'✅ Sí' if source['gratis'] else '❌ No'}")
        print()
    
    print("💡 CONSEJOS:")
    print("   1. Roboflow Universe es la mejor opción - datasets ya en formato YOLO")
    print("   2. Busca datasets con al menos 100-200 imágenes etiquetadas")
    print("   3. Prefiere datasets con múltiples clases (knife, gun, etc.)")
    print("   4. Verifica la calidad de las etiquetas antes de usar")
    print()


def download_roboflow_instructions():
    """Instrucciones para descargar desde Roboflow"""
    print("\n" + "=" * 70)
    print("INSTRUCCIONES PARA DESCARGAR DESDE ROBOFLOW")
    print("=" * 70)
    print()
    print("1. Ve a https://universe.roboflow.com/")
    print("2. Busca 'knife detection' o 'weapon detection'")
    print("3. Selecciona un dataset que te guste")
    print("4. Haz clic en 'Download'")
    print("5. Selecciona formato: 'YOLOv8'")
    print("6. Descarga el dataset")
    print("7. Descomprime el archivo ZIP")
    print("8. Ejecuta: python prepare_dataset.py <ruta_al_dataset>")
    print()
    print("Ejemplo de estructura descargada:")
    print("dataset-download/")
    print("  ├── train/")
    print("  │   ├── images/")
    print("  │   └── labels/")
    print("  ├── valid/")
    print("  │   ├── images/")
    print("  │   └── labels/")
    print("  ├── test/")
    print("  │   ├── images/")
    print("  │   └── labels/")
    print("  └── data.yaml")
    print()


def create_minimal_dataset_structure():
    """Crea estructura mínima para comenzar a crear tu propio dataset"""
    print("\n" + "=" * 70)
    print("CREANDO ESTRUCTURA PARA TU PROPIO DATASET")
    print("=" * 70)
    print()
    
    base_dir = Path("dataset")
    directories = [
        base_dir / "train" / "images",
        base_dir / "train" / "labels",
        base_dir / "val" / "images",
        base_dir / "val" / "labels",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Creado: {directory}")
    
    # Crear README con instrucciones
    readme_content = """# Dataset de Entrenamiento - Detección de Cuchillos

## Estructura
- train/images/: Imágenes de entrenamiento
- train/labels/: Etiquetas YOLO (.txt) correspondientes
- val/images/: Imágenes de validación
- val/labels/: Etiquetas YOLO (.txt) correspondientes

## Formato de Etiquetas YOLO
Cada archivo .txt debe tener el mismo nombre que su imagen correspondiente.

Formato por línea:
```
class_id center_x center_y width height
```

Ejemplo (cuchillo en el centro de la imagen):
```
0 0.5 0.5 0.2 0.3
```

Donde:
- class_id: 0 para knife, 1 para gun, 2 para weapon
- center_x, center_y: Coordenadas del centro normalizadas (0-1)
- width, height: Ancho y alto normalizados (0-1)

## Herramientas Recomendadas
1. LabelMe (pip install labelme): Para etiquetar imágenes manualmente
2. Roboflow: Para descargar datasets públicos
3. prepare_dataset.py: Para organizar tu dataset

## Mínimo Recomendado
- Train: 100-200 imágenes etiquetadas
- Val: 20-50 imágenes etiquetadas
"""
    
    readme_path = base_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"\n✅ README creado: {readme_path}")
    print("\n💡 Próximos pasos:")
    print("   1. Coloca tus imágenes en train/images/")
    print("   2. Etiquétalas usando LabelMe: pip install labelme && labelme")
    print("   3. Ejecuta: python prepare_dataset.py para organizar")
    print("   4. Ejecuta: python train.py para entrenar")
    print()


def create_sample_labelme_config():
    """Crea configuración de ejemplo para LabelMe"""
    config_content = """{
  "flags": {},
  "shapes": [],
  "version": "5.0.1",
  "imagePath": "image.jpg",
  "imageData": null,
  "imageHeight": 720,
  "imageWidth": 1280
}
"""
    
    config_path = Path("labelme_example.json")
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Archivo de ejemplo LabelMe creado: {config_path}")
    print("\n💡 Usa LabelMe para etiquetar:")
    print("   1. Instala: pip install labelme")
    print("   2. Ejecuta: labelme")
    print("   3. Abre tus imágenes")
    print("   4. Dibuja rectángulos alrededor de cuchillos")
    print("   5. Etiqueta como 'knife'")
    print("   6. Guarda las anotaciones")
    print("   7. Ejecuta: python prepare_dataset.py raw_data")
    print()


if __name__ == "__main__":
    import sys
    
    print("\n🔪 PREPARACIÓN DE DATASET PARA DETECCIÓN DE CUCHILLOS\n")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--sources":
            print_dataset_sources()
        elif sys.argv[1] == "--roboflow":
            download_roboflow_instructions()
        elif sys.argv[1] == "--create-structure":
            create_minimal_dataset_structure()
        elif sys.argv[1] == "--labelme":
            create_sample_labelme_config()
        else:
            print("Opciones disponibles:")
            print("  --sources          : Mostrar fuentes de datasets")
            print("  --roboflow         : Instrucciones para Roboflow")
            print("  --create-structure : Crear estructura básica")
            print("  --labelme          : Crear ejemplo LabelMe")
    else:
        print_dataset_sources()
        download_roboflow_instructions()
        print("\n¿Deseas crear la estructura básica? (s/n): ", end="")
        response = input().lower()
        if response == 's':
            create_minimal_dataset_structure()

