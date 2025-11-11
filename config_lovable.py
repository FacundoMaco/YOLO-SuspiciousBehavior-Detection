"""
Configuración para integración con Lovable/Supabase
"""

import os

# ============================================================================
# API KEY GENERADO AUTOMÁTICAMENTE
# ⚠️ IMPORTANTE: Este mismo valor debe estar en Supabase Secrets como CV_API_KEY
# ============================================================================
CV_API_KEY = "6lrke3If_wVdO1jQdbGVxJMNyb-u6UGMH6Vj4JceY1I"

# ============================================================================
# CONFIGURACIÓN DE SUPABASE/LOVABLE
# ============================================================================

# URL de tu Edge Function en Supabase
# ✅ CONFIGURADO con tu URL real
LOVABLE_API_URL = os.getenv(
    'LOVABLE_API_URL', 
    'https://zfmhmrbjxlrrmebtlpfx.supabase.co/functions/v1/receive-cv-alert'
)

# API Key para autenticación (mismo valor que CV_API_KEY)
LOVABLE_API_KEY = os.getenv('LOVABLE_API_KEY', CV_API_KEY)

# ============================================================================
# ✅ CONFIGURACIÓN COMPLETA
# ============================================================================
"""
✅ URL de Supabase configurada: https://zfmhmrbjxlrrmebtlpfx.supabase.co/functions/v1/receive-cv-alert
✅ API Key configurado: 6lrke3If_wVdO1jQdbGVxJMNyb-u6UGMH6Vj4JceY1I

El sistema está listo para enviar alertas a tu app de Lovable.

Cuando detecte armas o actividades delictivas/anómalas, las alertas aparecerán
instantáneamente en el AuthorityDashboard sin necesidad de recargar la página.
"""

# Función helper para obtener configuración
def get_config():
    """Retorna la configuración actual"""
    return {
        'cv_api_key': CV_API_KEY,
        'lovable_api_url': LOVABLE_API_URL,
        'lovable_api_key': LOVABLE_API_KEY
    }

