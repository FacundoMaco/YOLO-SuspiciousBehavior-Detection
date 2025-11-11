# 🔑 CONFIGURACIÓN PARA LOVABLE/SUPABASE

## API KEY GENERADO

**Copia este valor y pégalo en Lovable (Supabase Secrets):**

```
6lrke3If_wVdO1jQdbGVxJMNyb-u6UGMH6Vj4JceY1I
```

## 📋 PASOS PARA CONFIGURAR

### Paso 1: Configurar en Lovable/Supabase

1. Ve a tu proyecto en **Supabase Dashboard**
2. Ve a **Settings → Edge Functions → Secrets**
3. En el campo "Add Secret", escribe: `CV_API_KEY`
4. En el valor, pega: `6lrke3If_wVdO1jQdbGVxJMNyb-u6UGMH6Vj4JceY1I`
5. Haz clic en **Submit**

### Paso 2: Obtener tu URL de Supabase

1. En Supabase Dashboard, ve a **Settings → API**
2. Copia tu **Project URL** (algo como: `https://abcdefghijklmnop.supabase.co`)
3. Agrega al final: `/functions/v1/receive-cv-alert`
4. URL completa será: `https://abcdefghijklmnop.supabase.co/functions/v1/receive-cv-alert`

### Paso 3: Configurar en el sistema de Computer Vision

**Opción A: Editar `config_lovable.py`**

Abre `config_lovable.py` y reemplaza:
```python
LOVABLE_API_URL = 'https://TU_PROYECTO.supabase.co/functions/v1/receive-cv-alert'
```

Por tu URL real:
```python
LOVABLE_API_URL = 'https://abcdefghijklmnop.supabase.co/functions/v1/receive-cv-alert'
```

**Opción B: Usar variable de entorno**

```bash
export LOVABLE_API_URL="https://tu-proyecto.supabase.co/functions/v1/receive-cv-alert"
export LOVABLE_API_KEY="6lrke3If_wVdO1jQdbGVxJMNyb-u6UGMH6Vj4JceY1I"
```

## ✅ VERIFICACIÓN

Cuando ejecutes `python main.py`, deberías ver:

```
📋 Configuración cargada desde config_lovable.py
📡 Configurado para enviar alertas a: https://tu-proyecto.supabase.co/functions/v1/receive-cv-alert
🔑 API Key configurado: ********************...ceY1I
```

## 🚨 IMPORTANTE

- **Mismo API Key en ambos lados**: El valor `6lrke3If_wVdO1jQdbGVxJMNyb-u6UGMH6Vj4JceY1I` debe estar:
  - En Supabase Secrets como `CV_API_KEY`
  - En `config_lovable.py` como `LOVABLE_API_KEY` (o variable de entorno)

- **No compartas este API key públicamente**: Es un secreto de seguridad

## 📝 RESUMEN RÁPIDO

1. ✅ API Key: `6lrke3If_wVdO1jQdbGVxJMNyb-u6UGMH6Vj4JceY1I`
2. ✅ Configúralo en Supabase Secrets como `CV_API_KEY`
3. ✅ Edita `config_lovable.py` con tu URL de Supabase
4. ✅ ¡Listo! El sistema enviará alertas automáticamente

