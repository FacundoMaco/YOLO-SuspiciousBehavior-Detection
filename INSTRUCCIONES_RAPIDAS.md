# 🚀 CONFIGURACIÓN RÁPIDA - INTEGRACIÓN CON LOVABLE

## ✅ API KEY GENERADO

**Copia este valor exacto:**

```
6lrke3If_wVdO1jQdbGVxJMNyb-u6UGMH6Vj4JceY1I
```

---

## 📋 PASO 1: Configurar en Lovable (AHORA MISMO)

1. **En Lovable**, donde te está pidiendo el API key:
   - Campo: `CV_API_KEY`
   - Valor: `6lrke3If_wVdO1jQdbGVxJMNyb-u6UGMH6Vj4JceY1I`
   - Haz clic en **Submit**

✅ **Listo para Lovable**

---

## 📋 PASO 2: Obtener tu URL de Supabase

1. Ve a tu proyecto en **Supabase Dashboard**
2. **Settings → API**
3. Copia tu **Project URL** (ejemplo: `https://abcdefghijklmnop.supabase.co`)
4. Agrega al final: `/functions/v1/receive-cv-alert`
5. **URL completa será:** `https://abcdefghijklmnop.supabase.co/functions/v1/receive-cv-alert`

---

## 📋 PASO 3: Configurar en el Sistema de Computer Vision

**Edita el archivo `config_lovable.py`:**

Busca esta línea:
```python
LOVABLE_API_URL = 'https://TU_PROYECTO.supabase.co/functions/v1/receive-cv-alert'
```

Y reemplázala con tu URL real:
```python
LOVABLE_API_URL = 'https://TU_PROYECTO_REAL.supabase.co/functions/v1/receive-cv-alert'
```

**Ejemplo:**
```python
LOVABLE_API_URL = 'https://abcdefghijklmnop.supabase.co/functions/v1/receive-cv-alert'
```

---

## ✅ VERIFICACIÓN

Cuando ejecutes `python main.py`, deberías ver:

```
📋 Configuración cargada desde config_lovable.py
📡 Configurado para enviar alertas a: https://tu-proyecto.supabase.co/functions/v1/receive-cv-alert
🔑 API Key configurado: ********************...ceY1I
```

---

## 🎯 RESUMEN

1. ✅ **API Key en Lovable:** `6lrke3If_wVdO1jQdbGVxJMNyb-u6UGMH6Vj4JceY1I`
2. ⏳ **Obtener URL de Supabase** (Settings → API → Project URL + `/functions/v1/receive-cv-alert`)
3. ⏳ **Editar `config_lovable.py`** con tu URL real

---

## 🔄 FORMATO DE DATOS

El sistema transforma automáticamente los datos al formato que espera Supabase:

**Lo que envía el sistema CV:**
```json
{
  "activity": "hurto",
  "risk_level": "delictiva",
  "has_weapon": true,
  "confidence": 0.95,
  "location": "Av. Primavera 1234"
}
```

**Se transforma automáticamente a:**
```json
{
  "alert": {
    "camera_id": "CAM-SUR-1",
    "alert_type": "weapon",
    "confidence_score": 0.95,
    "location": "Av. Primavera 1234",
    "district": "Santiago de Surco",
    "detected_at": "2024-01-01T12:00:00Z"
  }
}
```

---

## 🚨 IMPORTANTE

- El mismo API key (`6lrke3If_wVdO1jQdbGVxJMNyb-u6UGMH6Vj4JceY1I`) debe estar:
  - ✅ En Supabase Secrets como `CV_API_KEY`
  - ✅ En `config_lovable.py` como `LOVABLE_API_KEY` (ya está configurado)

¡Todo está listo! Solo necesitas:
1. Pegar el API key en Lovable ✅
2. Configurar tu URL de Supabase en `config_lovable.py` ⏳

