"""
Script mejorado para descargar y preparar dataset de cuchillos desde Roboflow
Versión simplificada y más robusta
"""

import os
import zipfile
import shutil
from pathlib import Path

def download_roboflow_dataset():
    """Descarga y prepara un dataset público de cuchillos desde Roboflow"""
    
    print("=" * 70)
    print("DESCARGAR DATASET DE CUCHILLOS DESDE ROBOFLOW")
    print("=" * 70)
    print()
    print("📋 INSTRUCCIONES:")
    print("   1. En la página de Roboflow, haz clic en 'Use this Dataset'")
    print("   2. Selecciona 'Download' en el menú")
    print("   3. Elige formato: 'YOLOv8' o 'YOLOv5/YOLOv8'")
    print("   4. Elige tamaño: Small/Medium/Large (recomendado: Medium)")
    print("   5. Descarga el archivo ZIP")
    print()
    
    zip_path = input("Ingresa la ruta al archivo ZIP descargado\n(o presiona Enter para buscar 'dataset.zip' en el directorio actual): ").strip()
    
    if zip_path == "":
        zip_path = "dataset.zip"
    
    # Buscar archivos ZIP en el directorio actual si no se encuentra
    if not os.path.exists(zip_path):
        print(f"\n⚠️  No se encontró: {zip_path}")
        print("   Buscando archivos ZIP en el directorio actual...")
        
        zip_files = list(Path(".").glob("*.zip"))
        if zip_files:
            print("\n   Archivos ZIP encontrados:")
            for i, zf in enumerate(zip_files, 1):
                print(f"   {i}. {zf}")
            
            seleccion = input("\n   Selecciona el número del archivo (o Enter para cancelar): ").strip()
            if seleccion.isdigit() and 1 <= int(seleccion) <= len(zip_files):
                zip_path = str(zip_files[int(seleccion) - 1])
                print(f"   ✅ Seleccionado: {zip_path}")
            else:
                print("   ❌ Cancelado")
                return False
        else:
            print("\n   ❌ No se encontraron archivos ZIP")
            print("\n💡 OPCIONES:")
            print("   1. Descarga el dataset manualmente desde Roboflow")
            print("   2. Guarda el ZIP en este directorio")
            print("   3. Ejecuta este script nuevamente con la ruta completa")
            return False
    
    print(f"\n📦 Procesando: {zip_path}")
    
    # Crear directorio temporal
    temp_dir = Path("temp_dataset_extract")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Extraer archivo ZIP
        print("   Extrayendo archivo ZIP...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        print("   ✅ Archivo extraído")
        
        # Buscar estructura del dataset
        print("\n🔍 Buscando estructura del dataset...")
        
        # Buscar data.yaml primero (es el archivo clave)
        yaml_files = list(temp_dir.rglob("data.yaml")) + list(temp_dir.rglob("*.yaml"))
        
        if not yaml_files:
            print("   ⚠️  No se encontró data.yaml")
            print("   Buscando estructura manualmente...")
        else:
            yaml_file = yaml_files[0]
            print(f"   ✅ Encontrado: {yaml_file}")
            
            # Determinar directorio base
            base_dir = yaml_file.parent
            
            # Crear estructura objetivo
            dataset_dir = Path("dataset")
            dataset_dir.mkdir(exist_ok=True)
            
            # Copiar train
            train_imgs = base_dir / "train" / "images"
            train_labels = base_dir / "train" / "labels"
            
            if train_imgs.exists() and train_labels.exists():
                print("\n📋 Copiando archivos...")
                shutil.copytree(train_imgs, dataset_dir / "train" / "images", dirs_exist_ok=True)
                shutil.copytree(train_labels, dataset_dir / "train" / "labels", dirs_exist_ok=True)
                print(f"   ✅ Train: {len(list((dataset_dir / 'train' / 'images').glob('*')))} imágenes")
            
            # Copiar val/valid
            for val_name in ["valid", "val"]:
                val_imgs = base_dir / val_name / "images"
                val_labels = base_dir / val_name / "labels"
                
                if val_imgs.exists() and val_labels.exists():
                    shutil.copytree(val_imgs, dataset_dir / "val" / "images", dirs_exist_ok=True)
                    shutil.copytree(val_labels, dataset_dir / "val" / "labels", dirs_exist_ok=True)
                    print(f"   ✅ Val: {len(list((dataset_dir / 'val' / 'images').glob('*')))} imágenes")
                    break
            
            # Copiar data.yaml
            shutil.copy2(yaml_file, dataset_dir / "data.yaml")
            print(f"   ✅ data.yaml copiado")
            
            # Limpiar
            shutil.rmtree(temp_dir)
            
            print("\n" + "=" * 70)
            print("✅ DATASET PREPARADO EXITOSAMENTE")
            print("=" * 70)
            print(f"\n📁 Ubicación: {dataset_dir.absolute()}")
            print("\n💡 PRÓXIMOS PASOS:")
            print("   1. Verifica el dataset:")
            print("      python train.py --check")
            print("   2. Entrena el modelo:")
            print("      python train.py")
            print("   3. El modelo entrenado se guardará como 'best.pt'")
            print("   4. main.py lo detectará automáticamente")
            print()
            
            return True
        
        # Si no se encontró yaml, buscar estructura manualmente
        print("\n   Buscando estructura estándar...")
        for item in temp_dir.iterdir():
            if item.is_dir():
                train_path = item / "train"
                if train_path.exists():
                    print(f"   Encontrado: {item}")
        
        print("\n⚠️  Estructura no estándar detectada")
        print("   El dataset puede necesitar preparación manual")
        print("   Usa: python prepare_dataset.py <ruta_al_dataset>")
        
        return False
        
    except zipfile.BadZipFile:
        print(f"\n❌ Error: El archivo {zip_path} no es un ZIP válido")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_basic_yaml():
    """Crea un data.yaml básico"""
    yaml_content = """path: ./dataset
train: train/images
val: val/images

names:
  0: knife
  1: gun
  2: weapon
"""
    
    yaml_path = Path("dataset/data.yaml")
    yaml_path.parent.mkdir(exist_ok=True)
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"   ✅ data.yaml básico creado: {yaml_path}")


if __name__ == "__main__":
    download_roboflow_dataset()
