# Instrucciones de Configuración Rápida

## Requisitos Previos

1. Python 3.8 o superior
2. Git instalado

## Pasos para Configurar

### 1. Clonar/Actualizar el Repositorio

```bash
git pull origin main
```

### 2. Instalar Dependencias

```bash
pip3 install -r requirements.txt
```

### 3. Configurar API Key de Roboflow

El sistema usa el modelo de Roboflow para detectar cuchillos. Necesitas configurar tu API key:

#### Opción A: Variable de entorno (temporal)
```bash
export ROBOFLOW_API_KEY="tu-private-api-key-aqui"
python3 main.py
```

#### Opción B: Variable de entorno permanente
```bash
# En macOS/Linux (zsh)
echo 'export ROBOFLOW_API_KEY="tu-private-api-key-aqui"' >> ~/.zshrc
source ~/.zshrc

# Luego ejecutar
python3 main.py
```

#### Opción C: Windows (PowerShell)
```powershell
$env:ROBOFLOW_API_KEY="tu-private-api-key-aqui"
python main.py
```

### 4. Obtener tu API Key de Roboflow

1. Ve a https://roboflow.com/
2. Inicia sesión
3. Ve a **Settings** → **API Keys**
4. Copia tu **Private API Key** (no la Publishable)
5. Úsala en el paso 3

## Ejecutar el Sistema

```bash
python3 main.py
```

El sistema:
- Cargará automáticamente el modelo de Roboflow (`my-first-project-lchlk/1`)
- Detectará personas con pose estimation
- Detectar cuchillos y los marcará en **rojo** como "ARMA"
- Mostrará estadísticas en pantalla (Personas, Objetos, Armas)

## Notas Importantes

- **No subas tu API key a GitHub** - está protegida por `.gitignore`
- Si no configuras la API key, el sistema usará detección básica (knife, scissors) del modelo general
- Presiona 'q' para salir del sistema

## Solución de Problemas

### Error: "No module named 'roboflow'"
```bash
pip3 install roboflow
```

### Error: "ROBOFLOW_API_KEY no configurada"
Configura la variable de entorno como se indica en el paso 3.

### El modelo no detecta cuchillos
- Verifica que la API key esté correctamente configurada
- Asegúrate de que el modelo en Roboflow esté publicado y activo
- Verifica que la cámara tenga buena iluminación y el cuchillo sea visible

