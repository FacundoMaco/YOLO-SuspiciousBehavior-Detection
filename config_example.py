# Configuración para integración con app de Lovable

# Para usar esta configuración, crea un archivo .env o exporta las variables de entorno:
# export LOVABLE_API_URL="https://tu-app.lovable.app/api/alerts"
# export LOVABLE_API_KEY="tu-api-key-opcional"

# O modifica directamente estos valores:
LOVABLE_API_URL = None  # URL de tu endpoint en Lovable (ej: "https://tu-app.lovable.app/api/alerts")
LOVABLE_API_KEY = None  # API key opcional para autenticación

# Ejemplo de uso:
# LOVABLE_API_URL = "https://mi-app-123.lovable.app/api/alerts"
# LOVABLE_API_KEY = "sk_1234567890abcdef"

# Para usar esta configuración, descomenta las líneas y ajusta los valores
# Luego en main.py, cambia:
# lovable_api_url = os.getenv('LOVABLE_API_URL', None)
# Por:
# from config import LOVABLE_API_URL, LOVABLE_API_KEY
# lovable_api_url = LOVABLE_API_URL
# lovable_api_key = LOVABLE_API_KEY

