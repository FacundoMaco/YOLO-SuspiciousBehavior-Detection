# 🔍 DIAGNÓSTICO: Alertas se envían pero no aparecen en Lovable

## ✅ Lo que está funcionando

- ✅ El sistema CV envía alertas correctamente
- ✅ El Edge Function recibe las alertas (Status 201)
- ✅ Las alertas se insertan en la base de datos Supabase
- ✅ El formato de datos es correcto

## 🔍 Posibles problemas en Lovable

### 1. Verificar que la tabla `cv_alerts` existe y tiene Realtime habilitado

**En Supabase Dashboard:**

1. Ve a **Database → Tables**
2. Busca la tabla `cv_alerts`
3. Verifica que existe y tiene esta estructura:
   ```sql
   CREATE TABLE cv_alerts (
     id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
     camera_id text NOT NULL,
     alert_type text NOT NULL,
     confidence_score real NOT NULL,
     location text NOT NULL,
     district text NOT NULL,
     detected_at timestamptz DEFAULT now(),
     video_frame_url text,
     metadata jsonb,
     status text DEFAULT 'active',
     created_at timestamptz DEFAULT now()
   );
   ```

4. **CRÍTICO:** Verifica que Realtime está habilitado:
   - Ve a **Database → Replication**
   - Busca `cv_alerts` en la lista
   - Debe estar marcada como **Enabled**
   - Si no está, haz clic en **Enable**

### 2. Verificar el hook `useCVAlerts.ts`

**En tu código de Lovable, verifica que el hook esté así:**

```typescript
import { useEffect, useState } from 'react'
import { supabase } from '@/lib/supabase'

interface CVAlert {
  id: string
  camera_id: string
  alert_type: 'weapon' | 'violence' | 'theft' | 'suspicious'
  confidence_score: number
  location: string
  district: string
  detected_at: string
  video_frame_url: string | null
  metadata: Record<string, any>
  status: 'active' | 'dispatched' | 'resolved'
}

export function useCVAlerts() {
  const [alerts, setAlerts] = useState<CVAlert[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    // Cargar alertas iniciales
    const loadInitialAlerts = async () => {
      try {
        const { data, error } = await supabase
          .from('cv_alerts')
          .select('*')
          .eq('status', 'active')
          .order('detected_at', { ascending: false })
          .limit(50)

        if (error) throw error
        setAlerts(data || [])
        setLoading(false)
      } catch (err) {
        setError(err as Error)
        setLoading(false)
      }
    }

    loadInitialAlerts()

    // ⚠️ CRÍTICO: Suscripción Realtime
    const channel = supabase
      .channel('cv_alerts_changes')
      .on(
        'postgres_changes',
        {
          event: '*',  // Escuchar INSERT, UPDATE, DELETE
          schema: 'public',
          table: 'cv_alerts',
          filter: 'status=eq.active'  // Solo alertas activas
        },
        (payload) => {
          console.log('🔔 Nueva alerta recibida:', payload)
          
          if (payload.eventType === 'INSERT') {
            const newAlert = payload.new as CVAlert
            setAlerts((prev) => [newAlert, ...prev].slice(0, 50))
          } else if (payload.eventType === 'UPDATE') {
            const updated = payload.new as CVAlert
            if (updated.status !== 'active') {
              setAlerts((prev) => prev.filter((a) => a.id !== updated.id))
            } else {
              // Actualizar alerta existente
              setAlerts((prev) => 
                prev.map((a) => a.id === updated.id ? updated : a)
              )
            }
          }
        }
      )
      .subscribe((status) => {
        console.log('📡 Estado de suscripción Realtime:', status)
      })

    return () => {
      supabase.removeChannel(channel)
    }
  }, [])

  return { alerts, loading, error }
}
```

### 3. Verificar que `AuthorityDashboard.tsx` usa el hook

**Asegúrate de que esté usando `useCVAlerts()`:**

```typescript
import { useCVAlerts } from '@/hooks/useCVAlerts'

// Dentro del componente:
const { alerts: cvAlerts, loading: cvLoading } = useCVAlerts()

// Transformar cvAlerts al formato Alert
const transformedCVAlerts: Alert[] = cvAlerts.map((cvAlert) => ({
  id: parseInt(cvAlert.id.replace(/-/g, '').substring(0, 10), 16),
  type: 'AI_CAMERA',
  title: getAlertTitle(cvAlert.alert_type),
  location: cvAlert.location,
  district: cvAlert.district,
  time: new Date(cvAlert.detected_at).toLocaleString('es-PE'),
  priority: getPriority(cvAlert.alert_type),
  description: `Detección con ${(cvAlert.confidence_score * 100).toFixed(0)}% de confianza`,
  details: {
    cameraId: cvAlert.camera_id,
    confidence: cvAlert.confidence_score
  }
}))

// Combinar todas las alertas
const allAlerts = [
  ...transformedCVAlerts,
  ...mockPanicAlerts,
  ...mockCivilReports
]
```

### 4. Verificar en la consola del navegador

**Abre las DevTools (F12) y busca:**

1. **Errores en la consola:**
   - Busca mensajes en rojo
   - Verifica errores de conexión a Supabase

2. **Logs de Realtime:**
   - Deberías ver: `📡 Estado de suscripción Realtime: SUBSCRIBED`
   - Cuando llegue una alerta: `🔔 Nueva alerta recibida: {...}`

3. **Verificar conexión a Supabase:**
   ```typescript
   // En la consola del navegador:
   supabase.from('cv_alerts').select('*').limit(5)
     .then(console.log)
   ```

### 5. Verificar directamente en Supabase

**En Supabase Dashboard:**

1. Ve a **Database → Table Editor**
2. Selecciona la tabla `cv_alerts`
3. Verifica que las alertas se están insertando:
   - Deberías ver filas nuevas cuando el sistema CV detecta algo
   - Verifica que `status = 'active'`

### 6. Verificar RLS (Row Level Security)

**Si tienes RLS habilitado:**

1. Ve a **Authentication → Policies**
2. Busca políticas para `cv_alerts`
3. Asegúrate de que los usuarios con rol `authority` puedan leer:
   ```sql
   CREATE POLICY "Authorities can view cv_alerts"
   ON cv_alerts FOR SELECT
   USING (
     EXISTS (
       SELECT 1 FROM user_roles 
       WHERE user_id = auth.uid() 
       AND app_role = 'authority'
     )
   );
   ```

## 🧪 Prueba rápida

**Ejecuta esto en la consola del navegador de Lovable:**

```javascript
// 1. Verificar conexión
const { data, error } = await supabase
  .from('cv_alerts')
  .select('*')
  .eq('status', 'active')
  .limit(5)

console.log('Alertas en BD:', data)
console.log('Error:', error)

// 2. Verificar suscripción
const channel = supabase
  .channel('test_channel')
  .on('postgres_changes', 
    { event: '*', schema: 'public', table: 'cv_alerts' },
    (payload) => console.log('🔔 Alerta recibida:', payload)
  )
  .subscribe()

console.log('Canal suscrito:', channel.state)
```

## 📋 Checklist de verificación

- [ ] Tabla `cv_alerts` existe en Supabase
- [ ] Realtime está habilitado para `cv_alerts`
- [ ] Hook `useCVAlerts` está implementado correctamente
- [ ] `AuthorityDashboard` usa `useCVAlerts()`
- [ ] No hay errores en la consola del navegador
- [ ] La suscripción Realtime muestra `SUBSCRIBED`
- [ ] Las alertas aparecen en Supabase Table Editor
- [ ] RLS permite leer las alertas (si está habilitado)

## 🚨 Si nada funciona

**Último recurso - Forzar actualización manual:**

```typescript
// En AuthorityDashboard, agregar un botón de "Refresh"
const refreshAlerts = async () => {
  const { data } = await supabase
    .from('cv_alerts')
    .select('*')
    .eq('status', 'active')
    .order('detected_at', { ascending: false })
  
  // Actualizar estado manualmente
  setAlerts(data || [])
}
```

---

**¿Qué verificar primero?**

1. Abre la consola del navegador (F12)
2. Busca mensajes de error
3. Verifica que la suscripción Realtime esté `SUBSCRIBED`
4. Revisa si las alertas aparecen en Supabase Table Editor

