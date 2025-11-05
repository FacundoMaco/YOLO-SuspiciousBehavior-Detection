#!/usr/bin/env python3
"""
Script rápido para organizar el dataset descargado de Roboflow
"""

import shutil
from pathlib import Path

# Crear estructura
dataset_dir = Path("dataset")
dataset_dir.mkdir(exist_ok=True)

# Copiar train
if Path("train").exists():
    shutil.copytree("train", dataset_dir / "train", dirs_exist_ok=True)
    print("✅ Train copiado")

# Copiar valid a val
if Path("valid").exists():
    shutil.copytree("valid", dataset_dir / "val", dirs_exist_ok=True)
    print("✅ Valid copiado como val")

# Crear data.yaml corregido
yaml_content = """path: ./dataset
train: train/images
val: val/images

names:
  0: knife
"""

with open(dataset_dir / "data.yaml", 'w') as f:
    f.write(yaml_content)
print("✅ data.yaml creado")

# Contar archivos
train_imgs = len(list((dataset_dir / "train" / "images").glob("*")))
train_labels = len(list((dataset_dir / "train" / "labels").glob("*")))
val_imgs = len(list((dataset_dir / "val" / "images").glob("*")))
val_labels = len(list((dataset_dir / "val" / "labels").glob("*")))

print(f"\n📊 Dataset preparado:")
print(f"   Train: {train_imgs} imágenes, {train_labels} etiquetas")
print(f"   Val: {val_imgs} imágenes, {val_labels} etiquetas")
print(f"\n✅ Listo para entrenar! Ejecuta: python train.py")

