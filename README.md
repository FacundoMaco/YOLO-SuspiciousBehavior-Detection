# Sistema de Detección de Comportamientos Sospechosos con YOLOv8

Sistema avanzado de visión por computadora que detecta comportamientos sospechosos, armas, cuchillos, forcejeos y merodeadores en tiempo real usando YOLOv8.

## 🚀 Características

- ✅ **Detección de Pose Humana Mejorada**: Visualización robusta de esqueletos y ejes corporales (sin bugs visuales)
- ⚠️ **Detección de Amenazas**: Armas, cuchillos y objetos peligrosos
- 🥊 **Detección de Forcejeos**: Identifica cuando personas están muy cerca
- 🚶 **Detección de Merodeadores**: Identifica personas que permanecen inmóviles
- 📊 **Optimizado para 60 FPS**: Rendimiento mejorado con procesamiento optimizado
- 🎯 **Modelo Personalizado**: Soporte para modelos entrenados personalmente para mejor detección de cuchillos

## 📋 Requisitos

```bash
pip install -r requirements.txt
```

## 🎮 Uso Básico

```bash
python main.py
```

El sistema:
1. Cargará los modelos YOLOv8
2. Te pedirá seleccionar cámara web o archivo de video
3. Mostrará detecciones en tiempo real con visualización mejorada
4. Presiona 'q' para salir

## 🎯 Entrenar Modelo Personalizado para Detectar Cuchillos

El modelo estándar de YOLOv8 puede confundir cuchillos con otros objetos. Para mejorar la precisión y detectar correctamente cuchillos:

### Opción 1: Usar Dataset Público (Recomendado para empezar)

```bash
# Ver fuentes de datasets públicos
python setup_dataset.py --sources

# Ver instrucciones para Roboflow (más fácil)
python setup_dataset.py --roboflow
```

**Pasos rápidos con Roboflow:**
1. Ve a https://universe.roboflow.com/
2. Busca "knife detection" o "weapon detection"
3. Descarga un dataset en formato YOLOv8
4. Descomprime y ejecuta: `python prepare_dataset.py <ruta_del_dataset>`
5. Ejecuta: `python train.py`

### Opción 2: Crear tu Propio Dataset

```bash
# 1. Crear estructura básica
python setup_dataset.py --create-structure

# 2. Instalar LabelMe para etiquetar
pip install labelme
labelme

# 3. Etiquetar tus imágenes:
#    - Abre tus imágenes en LabelMe
#    - Dibuja rectángulos alrededor de cuchillos
#    - Etiqueta como 'knife'
#    - Guarda las anotaciones

# 4. Organizar dataset
python prepare_dataset.py raw_data

# 5. Entrenar modelo
python train.py
```

### Estructura del Dataset

```
dataset/
├── train/
│   ├── images/    # Imágenes de entrenamiento
│   └── labels/    # Etiquetas YOLO (.txt)
├── val/
│   ├── images/    # Imágenes de validación
│   └── labels/    # Etiquetas YOLO (.txt)
└── data.yaml      # Configuración del dataset
```

### Verificar Dataset Antes de Entrenar

```bash
# Verificar estructura y contar anotaciones
python train.py --check
```

### Entrenar Modelo

```bash
python train.py
```

El entrenamiento:
- ✅ Verifica automáticamente la estructura del dataset
- ✅ Cuenta las anotaciones disponibles
- ✅ Usa YOLOv8n como modelo base
- ✅ Entrena por 100 épocas con early stopping
- ✅ Guarda el mejor modelo automáticamente
- ✅ Copia `best.pt` a la raíz del proyecto

El script `main.py` detectará automáticamente el modelo entrenado y lo usará para mejor detección de cuchillos.

## 🔧 Mejoras en Detección de Pose

Se han implementado mejoras significativas para evitar bugs visuales:

- ✅ **Validación completa de keypoints**: Verifica coordenadas válidas antes de dibujar
- ✅ **Filtrado por confianza**: Solo muestra puntos con confianza > 0.25
- ✅ **Validación de bounding boxes**: Evita errores con coordenadas inválidas
- ✅ **Manejo robusto de errores**: No interrumpe el flujo si hay problemas de visualización
- ✅ **Validación de dimensiones**: Verifica que todo esté dentro del frame

## 📊 Optimizaciones de Rendimiento

El sistema está optimizado para alcanzar 60 FPS mediante:

1. **Procesamiento a resolución reducida** (640px) mientras se mantiene la visualización original
2. **Skip frames** para detección de pose (reduce carga computacional)
3. **Configuración optimizada** de YOLOv8 (`verbose=False`, `imgsz` fijo)
4. **Control de FPS** para mantener tasa constante

### Ajustar Rendimiento

En `main.py`, puedes modificar:

```python
PROCESS_RESOLUTION = 640  # Reducir a 320 para más FPS (menos precisión)
SKIP_FRAMES = 1          # Procesar pose cada N frames
TARGET_FPS = 60          # FPS objetivo
```

## 🎨 Visualización

- **Verde**: Personas detectadas
- **Rojo**: Amenazas (armas, cuchillos)
- **Amarillo**: Otros objetos
- **Azul**: Líneas del esqueleto (sin bugs visuales)
- **Amarillo claro**: Puntos de articulación
- **Rojo**: Eje central de la persona

## 📝 Configuración de Detección

Ajusta estos parámetros en `main.py` según tus necesidades:

```python
MIN_FIGHT_DISTANCE = 100      # Distancia para considerar forcejeo (píxeles)
STALLING_TIME = 5             # Tiempo para considerar merodeador (segundos)
MAX_SPEED_NORMAL = 50         # Velocidad máxima normal (píxeles/frame)
```

## 🔧 Solución de Problemas

### Modelo no detecta cuchillos correctamente
- **Solución**: Entrena un modelo personalizado siguiendo los pasos arriba
- Asegúrate de tener suficientes imágenes etiquetadas (mínimo 100-200 por clase)
- Usa datasets públicos de Roboflow para empezar rápido
- Verifica que las etiquetas sean correctas

### Bugs visuales en esqueletos (líneas fuera de lugar)
- **Solucionado**: Las mejoras implementadas validan todos los keypoints antes de dibujar
- Si aún ves problemas, reduce `PROCESS_RESOLUTION` para mejor precisión de pose

### FPS bajo
- Reduce `PROCESS_RESOLUTION` a 320 o 480
- Aumenta `SKIP_FRAMES` a 2 o 3
- Usa GPU si está disponible (configura `device=0` en train.py)

### Error al cargar modelo personalizado
- Verifica que `best.pt` esté en la raíz del proyecto
- Asegúrate de que el modelo fue entrenado con las mismas clases que esperas
- Ejecuta `python train.py --check` para verificar el dataset

### Error durante entrenamiento
- Verifica que tengas al menos 50-100 imágenes etiquetadas
- Asegúrate de que las etiquetas estén en formato YOLO correcto
- Verifica que `data.yaml` tenga la estructura correcta
- Si no tienes GPU, cambia `device=0` a `device='cpu'` en train.py

## 📚 Scripts Disponibles

- `main.py`: Script principal de detección
- `train.py`: Entrenamiento del modelo personalizado
- `prepare_dataset.py`: Preparar y organizar datasets
- `setup_dataset.py`: Guía y herramientas para obtener datasets

## 📚 Recursos

- [Documentación YOLOv8](https://docs.ultralytics.com/)
- [LabelMe - Herramienta de etiquetado](https://github.com/labelmeai/labelme)
- [Roboflow Universe - Datasets públicos](https://universe.roboflow.com/)

## ⚠️ Consideraciones Éticas y Legales

- Este sistema es para fines educativos y de seguridad legítima
- Respeta las leyes locales sobre vigilancia y privacidad
- Usa solo imágenes legales y éticas para entrenamiento
- **No uses armas reales para pruebas** - usa armas de juguete u objetos simulados
- Para pruebas con cuchillos, usa cuchillos de cocina normales o objetos simulados

## 📄 Licencia

Proyecto educativo - Úsalo responsablemente

