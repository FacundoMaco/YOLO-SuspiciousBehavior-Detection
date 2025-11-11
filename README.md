# Sistema de Videovigilancia Inteligente para Detección de Actividades Anómalas

Sistema avanzado de visión por computadora basado en el artículo de Sathiyavathi et al. (2021) que detecta comportamientos humanos anómalos en tiempo real usando CNN, YOLOv8-pose, clasificación de riesgo y análisis temporal.

## 🚀 Características Principales

- ✅ **Detección de Pose Humana**: Extracción de 17 puntos clave corporales usando YOLOv8-pose
- 🧠 **Clasificación de Actividades con CNN**: Reconocimiento de 5 actividades (caminar, sentarse, interactuar, saludar, hurto)
- ⚠️ **Clasificación de Riesgo**: Sistema de 3 niveles (segura, anómala, delictiva)
- 📊 **Análisis Temporal**: Detección de patrones anómalos mediante análisis de secuencias
- 🗄️ **Base de Datos de Eventos**: Almacenamiento SQLite de todos los eventos detectados
- 🌐 **API REST**: Endpoints para recibir y consultar alertas en tiempo real
- ⚠️ **Detección de Amenazas**: Armas, cuchillos y objetos peligrosos

## 📋 Requisitos

```bash
pip install -r requirements.txt
```

### Dependencias Principales

- **YOLOv8** (Ultralytics): Detección de objetos y pose
- **TensorFlow/Keras**: Modelo CNN para clasificación de actividades
- **Flask**: API REST para alertas
- **SQLite**: Base de datos de eventos (incluido en Python)

## 🎮 Uso Básico

```bash
python main.py
```

El sistema:
1. Cargará los modelos YOLOv8 y el clasificador CNN de actividades
2. Iniciará la API REST en `http://localhost:5000`
3. Te pedirá seleccionar cámara web o archivo de video
4. Te pedirá la ubicación (opcional)
5. Mostrará detecciones en tiempo real con:
   - Esqueletos de personas detectadas
   - Actividades clasificadas
   - Niveles de riesgo (segura/anómala/delictiva)
   - Alertas automáticas para eventos críticos
6. Presiona 'q' para salir

## 🧠 Clasificación de Actividades

El sistema clasifica las siguientes actividades basándose en los keypoints de pose:

- **Caminar**: Movimiento normal de desplazamiento
- **Sentarse**: Persona en posición sentada
- **Interactuar**: Interacción entre personas
- **Saludar**: Gestos de saludo
- **Hurto**: Actividad delictiva detectada

## ⚠️ Clasificación de Riesgo

El sistema clasifica automáticamente el nivel de riesgo:

- **Segura** (Verde): Actividades normales sin amenazas
- **Anómala** (Naranja): Actividades sospechosas que requieren atención
- **Delictiva** (Rojo): Actividades delictivas o presencia de armas

## 📊 Análisis Temporal

El sistema analiza secuencias de video para detectar:
- Patrones de comportamiento inusuales
- Cambios bruscos en actividades
- Variabilidad en movimientos
- Secuencias delictivas

## 🗄️ Base de Datos

Todos los eventos se almacenan automáticamente en `database/events.db` con:
- Timestamp del evento
- Actividad detectada
- Nivel de riesgo
- Confianza de la detección
- Ubicación
- Keypoints de la persona
- Estado de alerta enviada

## 🌐 API REST

La API REST está disponible en `http://localhost:5000` con los siguientes endpoints:

### Endpoints Disponibles

- `GET /health` - Estado del servicio
- `POST /alerts` - Enviar una alerta
- `GET /alerts` - Obtener alertas (con filtros opcionales)
- `GET /alerts/<id>` - Obtener una alerta específica
- `DELETE /alerts/<id>` - Eliminar una alerta
- `GET /stats` - Estadísticas del sistema

### Ejemplo de Uso de la API

```bash
# Ver estado del servicio
curl http://localhost:5000/health

# Obtener alertas delictivas
curl http://localhost:5000/alerts?risk_level=delictiva

# Ver estadísticas
curl http://localhost:5000/stats
```

## 🎯 Entrenar Modelo de Clasificación de Actividades

Para entrenar el modelo CNN de clasificación de actividades:

### 1. Preparar Datos

Estructura de directorios esperada:

```
data/activities/
├── caminar/
│   ├── keypoints_001.npy
│   ├── keypoints_002.npy
│   └── ...
├── sentarse/
├── interactuar/
├── saludar/
└── hurto/
```

Cada archivo `.npy` debe contener keypoints de pose con forma `(17, 2)` o `(34,)`.

### 2. Entrenar Modelo

```bash
python train_activity_model.py data/activities 20 32
```

Parámetros:
- `data/activities`: Directorio con los datos
- `20`: Número de épocas
- `32`: Tamaño del batch

El modelo entrenado se guardará en `models/activity_model.h5` y se cargará automáticamente en `main.py`.

## 🎯 Entrenar Modelo Personalizado para Detectar Cuchillos

Para mejorar la detección de cuchillos y armas:

```bash
# Ver instrucciones
python setup_dataset.py --roboflow

# Preparar dataset
python prepare_dataset.py <ruta_del_dataset>

# Entrenar modelo
python train.py
```

## 📁 Estructura del Proyecto

```
YOLO-SuspiciousBehavior-Detection/
├── main.py                      # Script principal
├── train_activity_model.py      # Entrenamiento del modelo CNN
├── train.py                     # Entrenamiento de detección de cuchillos
├── models/
│   ├── activity_classifier.py   # Clasificador CNN de actividades
│   └── activity_model.h5        # Modelo entrenado (generado)
├── utils/
│   ├── risk_classifier.py       # Clasificador de riesgo
│   └── temporal_analyzer.py    # Análisis temporal
├── database/
│   ├── event_db.py              # Gestor de base de datos
│   └── events.db               # Base de datos SQLite (generada)
├── api/
│   └── alert_api.py             # API REST para alertas
├── DOCUMENTACION.md              # Documentación del artículo base
└── requirements.txt             # Dependencias
```

## 🎨 Visualización

El sistema muestra en pantalla:

- **Verde**: Personas con actividades seguras
- **Naranja**: Personas con actividades anómalas
- **Rojo**: Personas con actividades delictivas o armas
- **Azul**: Líneas del esqueleto
- **Amarillo claro**: Puntos de articulación
- **Rojo**: Eje central de la persona

## 📊 Estadísticas del Sistema

Al cerrar el sistema, se muestran estadísticas de:
- Total de eventos detectados
- Eventos por nivel de riesgo
- Confianza promedio
- Alertas enviadas

También disponibles en tiempo real vía API REST: `GET /stats`

## 🔧 Solución de Problemas

### Modelo de actividades no encontrado
- El sistema creará un modelo nuevo automáticamente
- Para mejor precisión, entrena con datos reales usando `train_activity_model.py`

### API REST no inicia
- Verifica que el puerto 5000 esté disponible
- Cambia el puerto en `main.py`: `AlertAPI(host='localhost', port=5001)`

### FPS bajo
- Reduce la resolución del video de entrada
- Usa GPU si está disponible (configura en TensorFlow)

### Error al cargar modelos YOLOv8
- Los modelos se descargarán automáticamente la primera vez
- Verifica tu conexión a internet

## 📚 Referencias

- Sathiyavathi, V., Jessey, M., Selvakumar, K., & SaiRamesh, L. (2021). Smart surveillance system for abnormal activity detection using CNN. In D. J. Hemanth (Ed.), Advances in Parallel Computing Technologies and Applications (pp. 341–349).

## ⚠️ Consideraciones Éticas y Legales

- Este sistema es para fines educativos y de seguridad legítima
- Respeta las leyes locales sobre vigilancia y privacidad
- Usa solo imágenes legales y éticas para entrenamiento
- **No uses armas reales para pruebas** - usa armas de juguete u objetos simulados
- Para pruebas con cuchillos, usa cuchillos de cocina normales o objetos simulados

## 📄 Licencia

Proyecto educativo - Úsalo responsablemente

## 👥 Integrantes

- Maria Fernanda Tapia Yepez
- Marianet Leon Astuhuaman
- Mariana Emy Sanchez Galdos
- Manuel Aarón Torres Tolentino
