"""
Script rápido para obtener y preparar dataset de cuchillos
"""

import os
from pathlib import Path

def setup_knife_dataset():
    """Guía paso a paso para obtener y preparar dataset de cuchillos"""
    
    print("=" * 70)
    print("CONFIGURACIÓN DE DATASET PARA DETECCIÓN DE CUCHILLOS")
    print("=" * 70)
    print()
    
    print("📋 PASOS PARA OBTENER EL DATASET:")
    print()
    print("1. Ve a: https://universe.roboflow.com/")
    print("2. Busca: 'knife detection' o 'weapon detection'")
    print("3. Selecciona un dataset (recomendado: >100 imágenes)")
    print("4. Haz clic en 'Download'")
    print("5. Selecciona formato: 'YOLOv8'")
    print("6. Descarga el archivo ZIP")
    print()
    
    respuesta = input("¿Ya descargaste el archivo ZIP? (s/n): ").lower()
    
    if respuesta == 's':
        zip_path = input("Ingresa la ruta al archivo ZIP [o presiona Enter para 'dataset.zip']: ").strip()
        
        if zip_path == "":
            zip_path = "dataset.zip"
        
        if os.path.exists(zip_path):
            print(f"\n✅ Archivo encontrado: {zip_path}")
            print("   Ejecuta: python download_dataset.py")
            print("   Luego: python train.py")
        else:
            print(f"\n❌ No se encontró: {zip_path}")
            print("   Asegúrate de que el archivo esté en el directorio actual")
    else:
        print("\n💡 ALTERNATIVA: Crear tu propio dataset")
        print()
        print("1. Instala LabelMe:")
        print("   pip install labelme")
        print()
        print("2. Ejecuta LabelMe:")
        print("   labelme")
        print()
        print("3. Etiqueta tus imágenes:")
        print("   - Abre tus imágenes")
        print("   - Dibuja rectángulos alrededor de cuchillos")
        print("   - Etiqueta como 'knife'")
        print("   - Guarda las anotaciones")
        print()
        print("4. Organiza el dataset:")
        print("   python prepare_dataset.py raw_data")
        print()
        print("5. Entrena el modelo:")
        print("   python train.py")
        print()
    
    print("\n📚 RECURSOS:")
    print("   - Roboflow Universe: https://universe.roboflow.com/")
    print("   - LabelMe: https://github.com/labelmeai/labelme")
    print()


if __name__ == "__main__":
    setup_knife_dataset()

