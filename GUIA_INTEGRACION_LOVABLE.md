# Guía de Integración con App de Lovable

Esta guía te ayudará a conectar el sistema de videovigilancia con tu app de Lovable para recibir notificaciones en tiempo real.

## 📋 Requisitos Previos

1. Tu app de Lovable debe tener un endpoint que reciba alertas POST
2. El endpoint debe aceptar JSON con el formato especificado abajo

## 🔧 Configuración

### Opción 1: Variables de Entorno (Recomendado)

```bash
# Configurar URL de tu app de Lovable
export LOVABLE_API_URL="https://tu-app.lovable.app/api/alerts"

# Opcional: Si tu app requiere autenticación
export LOVABLE_API_KEY="tu-api-key-aqui"
```

### Opción 2: Modificar main.py directamente

Edita `main.py` y cambia estas líneas (alrededor de la línea 71):

```python
# Cambiar de:
lovable_api_url = os.getenv('LOVABLE_API_URL', None)

# A:
lovable_api_url = "https://tu-app.lovable.app/api/alerts"
lovable_api_key = "tu-api-key-opcional"  # Si es necesario
```

## 📡 Formato de Datos que se Envía

Cuando el sistema detecta un evento crítico (arma o actividad delictiva/anómala), envía un POST a tu app con este formato:

```json
{
  "alert": {
    "activity": "hurto",
    "risk_level": "delictiva",
    "confidence": 0.95,
    "person_id": 1,
    "location": "Zona A",
    "has_weapon": true,
    "keypoints": [[x1, y1], [x2, y2], ...],
    "frame_image_path": "/path/to/image.jpg",
    "timestamp": "2024-01-01T12:00:00"
  },
  "source": "smart_surveillance_system",
  "version": "1.0"
}
```

### Campos Importantes:

- **activity**: Actividad detectada (`caminar`, `sentarse`, `interactuar`, `saludar`, `hurto`)
- **risk_level**: Nivel de riesgo (`segura`, `anómala`, `delictiva`)
- **confidence**: Confianza de la detección (0.0 a 1.0)
- **has_weapon**: `true` si se detectó un arma
- **location**: Ubicación del evento
- **timestamp**: Fecha y hora del evento

## 🎯 Configurar Endpoint en Lovable

### 1. Crear Endpoint en Lovable

En tu app de Lovable, crea un endpoint que reciba POST requests:

**Ejemplo de endpoint en Lovable:**

```javascript
// Endpoint: POST /api/alerts
export async function POST(request) {
  try {
    const data = await request.json();
    
    // Validar que tenga el campo 'alert'
    if (!data.alert) {
      return Response.json({ error: 'Invalid format' }, { status: 400 });
    }
    
    const alert = data.alert;
    
    // Procesar la alerta
    console.log('Alerta recibida:', alert);
    
    // Aquí puedes:
    // - Guardar en base de datos
    // - Enviar notificación push
    // - Enviar email
    // - Mostrar en dashboard
    
    // Ejemplo: Guardar en base de datos
    // await db.alerts.create({
    //   activity: alert.activity,
    //   risk_level: alert.risk_level,
    //   has_weapon: alert.has_weapon,
    //   location: alert.location,
    //   timestamp: alert.timestamp
    // });
    
    return Response.json({ 
      status: 'success',
      message: 'Alert received' 
    }, { status: 200 });
    
  } catch (error) {
    console.error('Error processing alert:', error);
    return Response.json({ error: 'Internal server error' }, { status: 500 });
  }
}
```

### 2. Autenticación (Opcional)

Si tu endpoint requiere autenticación, puedes usar:

**Opción A: Bearer Token**
```python
# En main.py o variable de entorno
export LOVABLE_API_KEY="tu-bearer-token"
```

El sistema enviará:
```
Authorization: Bearer tu-bearer-token
```

**Opción B: API Key en Header**
Si prefieres otro formato, modifica `api/alert_api.py` línea 323:
```python
headers['X-API-Key'] = self.external_api_key
```

## 🧪 Probar la Integración

### 1. Verificar que el sistema detecta la configuración

Al ejecutar `python main.py`, deberías ver:
```
📡 Configurado para enviar alertas a: https://tu-app.lovable.app/api/alerts
```

### 2. Probar manualmente con curl

```bash
curl -X POST https://tu-app.lovable.app/api/alerts \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer tu-api-key" \
  -d '{
    "alert": {
      "activity": "hurto",
      "risk_level": "delictiva",
      "confidence": 0.95,
      "has_weapon": true,
      "location": "Zona de prueba",
      "timestamp": "2024-01-01T12:00:00"
    },
    "source": "smart_surveillance_system",
    "version": "1.0"
  }'
```

### 3. Verificar en tiempo real

Cuando el sistema detecte un evento crítico, verás en la consola:
```
✅ Alerta enviada exitosamente a https://tu-app.lovable.app/api/alerts
```

## 🔍 Troubleshooting

### El sistema no envía alertas

1. **Verifica la URL**: Asegúrate de que `LOVABLE_API_URL` esté configurada correctamente
2. **Verifica la conexión**: Prueba hacer un curl manual al endpoint
3. **Revisa los logs**: El sistema mostrará errores si hay problemas de conexión

### Error de autenticación

1. Verifica que `LOVABLE_API_KEY` esté configurada correctamente
2. Verifica el formato de autenticación en tu endpoint de Lovable
3. Si usas otro formato, modifica `api/alert_api.py` línea 323

### Timeout o errores de conexión

1. Verifica que tu app de Lovable esté desplegada y accesible
2. Verifica que el endpoint acepte POST requests
3. Verifica que no haya firewall bloqueando la conexión

## 📝 Ejemplo Completo de Endpoint en Lovable

```javascript
// app/api/alerts/route.js (Next.js) o similar

import { NextResponse } from 'next/server';

export async function POST(request) {
  try {
    // Verificar autenticación (opcional)
    const authHeader = request.headers.get('authorization');
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }
    
    const data = await request.json();
    
    if (!data.alert) {
      return NextResponse.json(
        { error: 'Invalid format' },
        { status: 400 }
      );
    }
    
    const alert = data.alert;
    
    // Procesar según el nivel de riesgo
    if (alert.risk_level === 'delictiva') {
      // ALERTA CRÍTICA - Actuar inmediatamente
      console.log('🚨 ALERTA DELICTIVA:', alert);
      
      // Enviar notificación push
      // await sendPushNotification({
      //   title: 'Alerta de Seguridad',
      //   body: `Actividad delictiva detectada en ${alert.location}`,
      //   priority: 'high'
      // });
      
    } else if (alert.risk_level === 'anómala') {
      // ALERTA ANÓMALA - Monitorear
      console.log('⚠️ ALERTA ANÓMALA:', alert);
    }
    
    // Guardar en base de datos
    // await db.alerts.create({
    //   activity: alert.activity,
    //   risk_level: alert.risk_level,
    //   confidence: alert.confidence,
    //   has_weapon: alert.has_weapon,
    //   location: alert.location,
    //   person_id: alert.person_id,
    //   timestamp: new Date(alert.timestamp)
    // });
    
    return NextResponse.json({
      status: 'success',
      message: 'Alert received',
      alert_id: alert.person_id
    });
    
  } catch (error) {
    console.error('Error processing alert:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
```

## ✅ Checklist de Integración

- [ ] Endpoint creado en Lovable (`/api/alerts` o similar)
- [ ] Endpoint acepta POST requests
- [ ] Endpoint acepta JSON con formato correcto
- [ ] Variable `LOVABLE_API_URL` configurada
- [ ] Variable `LOVABLE_API_KEY` configurada (si es necesario)
- [ ] Probado con curl manualmente
- [ ] Sistema de videovigilancia configurado
- [ ] Probado con detección real

## 🚀 Listo!

Una vez configurado, cada vez que el sistema detecte:
- Una persona con arma → Envía alerta `delictiva`
- Una actividad anómala → Envía alerta `anómala`

Tu app de Lovable recibirá la notificación en tiempo real y podrás:
- Mostrarla en un dashboard
- Enviar notificaciones push
- Guardar en base de datos
- Activar alarmas
- Etc.

