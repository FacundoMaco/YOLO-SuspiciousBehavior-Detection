"""
API REST para envío de alertas
Basado en Flask para comunicación con sistemas externos
"""

from datetime import datetime
from typing import Dict, Optional
import threading
import json
import os

# Flask es opcional - solo se necesita si se quiere usar la API REST
try:
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None
    jsonify = None
    request = None
    CORS = None

# Requests para enviar alertas a endpoints externos
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None


class AlertAPI:
    """
    API REST para recibir y enviar alertas del sistema de videovigilancia
    """
    
    def __init__(self, host='localhost', port=5000, enable_cors=True, 
                 external_api_url=None, external_api_key=None):
        """
        Inicializa la API REST
        
        Args:
            host: Host donde escuchar
            port: Puerto donde escuchar
            enable_cors: Habilitar CORS para peticiones cross-origin
            external_api_url: URL de la app externa (Lovable) para enviar alertas
            external_api_key: API key opcional para autenticación
        """
        if not FLASK_AVAILABLE:
            print("⚠️  Flask no está disponible. La API REST no funcionará.")
            print("   Instala con: pip install flask flask-cors")
            self.app = None
            self.host = host
            self.port = port
            self.enable_cors = enable_cors
            self.pending_alerts = []
            self.alert_callback = None
            self.external_api_url = external_api_url
            self.external_api_key = external_api_key
            return
        
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.enable_cors = enable_cors
        
        if enable_cors:
            CORS(self.app)
        
        # Lista de alertas pendientes
        self.pending_alerts = []
        
        # Callback para cuando se recibe una alerta
        self.alert_callback = None
        
        # Configuración para enviar a app externa (Lovable)
        self.external_api_url = external_api_url or os.getenv('LOVABLE_API_URL', None)
        self.external_api_key = external_api_key or os.getenv('LOVABLE_API_KEY', None)
        
        if self.external_api_url:
            print(f"📡 Configurado para enviar alertas a: {self.external_api_url}")
        
        # Configurar rutas
        self.setup_routes()
    
    def setup_routes(self):
        """Configura las rutas de la API"""
        if not FLASK_AVAILABLE or self.app is None:
            return
        
        @self.app.route('/health', methods=['GET'])
        def health():
            """Endpoint de salud del servicio"""
            return jsonify({
                'status': 'ok',
                'service': 'Smart Surveillance Alert API',
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/alerts', methods=['POST'])
        def receive_alert():
            """
            Recibe una alerta del sistema de videovigilancia
            
            Body esperado:
            {
                "activity": "hurto",
                "risk_level": "delictiva",
                "confidence": 0.95,
                "person_id": 1,
                "location": "Zona A",
                "has_weapon": true,
                "timestamp": "2024-01-01T12:00:00",
                "keypoints": [...],
                "frame_image_path": "/path/to/image.jpg"
            }
            """
            try:
                data = request.get_json()
                
                if not data:
                    return jsonify({'error': 'No data provided'}), 400
                
                # Validar campos requeridos
                required_fields = ['activity', 'risk_level', 'confidence']
                for field in required_fields:
                    if field not in data:
                        return jsonify({'error': f'Missing required field: {field}'}), 400
                
                # Agregar timestamp si no existe
                if 'timestamp' not in data:
                    data['timestamp'] = datetime.now().isoformat()
                
                # Agregar a lista de alertas pendientes
                alert = {
                    'id': len(self.pending_alerts) + 1,
                    **data,
                    'received_at': datetime.now().isoformat()
                }
                self.pending_alerts.append(alert)
                
                # Llamar callback si existe
                if self.alert_callback:
                    self.alert_callback(alert)
                
                return jsonify({
                    'status': 'success',
                    'alert_id': alert['id'],
                    'message': 'Alert received successfully'
                }), 201
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/alerts', methods=['GET'])
        def get_alerts():
            """
            Obtiene las alertas pendientes
            
            Query params:
            - limit: Número máximo de alertas (default: 100)
            - risk_level: Filtrar por nivel de riesgo
            """
            try:
                limit = int(request.args.get('limit', 100))
                risk_level = request.args.get('risk_level', None)
                
                alerts = self.pending_alerts[-limit:] if limit > 0 else self.pending_alerts
                
                if risk_level:
                    alerts = [a for a in alerts if a.get('risk_level') == risk_level]
                
                return jsonify({
                    'status': 'success',
                    'count': len(alerts),
                    'alerts': alerts
                }), 200
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/alerts/<int:alert_id>', methods=['GET'])
        def get_alert(alert_id):
            """Obtiene una alerta específica por ID"""
            try:
                alert = next((a for a in self.pending_alerts if a['id'] == alert_id), None)
                
                if not alert:
                    return jsonify({'error': 'Alert not found'}), 404
                
                return jsonify({
                    'status': 'success',
                    'alert': alert
                }), 200
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/alerts/<int:alert_id>', methods=['DELETE'])
        def delete_alert(alert_id):
            """Elimina una alerta específica"""
            try:
                global_index = None
                for i, alert in enumerate(self.pending_alerts):
                    if alert['id'] == alert_id:
                        global_index = i
                        break
                
                if global_index is None:
                    return jsonify({'error': 'Alert not found'}), 404
                
                deleted_alert = self.pending_alerts.pop(global_index)
                
                return jsonify({
                    'status': 'success',
                    'message': 'Alert deleted',
                    'alert': deleted_alert
                }), 200
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/stats', methods=['GET'])
        def get_stats():
            """Obtiene estadísticas de las alertas"""
            try:
                total_alerts = len(self.pending_alerts)
                
                risk_levels = {}
                for alert in self.pending_alerts:
                    risk = alert.get('risk_level', 'unknown')
                    risk_levels[risk] = risk_levels.get(risk, 0) + 1
                
                return jsonify({
                    'status': 'success',
                    'total_alerts': total_alerts,
                    'risk_level_distribution': risk_levels
                }), 200
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def set_alert_callback(self, callback):
        """
        Establece un callback que se ejecutará cuando se reciba una alerta
        
        Args:
            callback: Función que recibe un dict con la alerta
        """
        self.alert_callback = callback
    
    def send_alert(self, activity: str, risk_level: str, confidence: float,
                   person_id: Optional[int] = None, location: Optional[str] = None,
                   has_weapon: bool = False, keypoints: Optional[list] = None,
                   frame_image_path: Optional[str] = None):
        """
        Envía una alerta a través de la API
        Si hay external_api_url configurado, envía a la app externa (Lovable)
        
        Args:
            activity: Actividad detectada
            risk_level: Nivel de riesgo
            confidence: Confianza de la detección
            person_id: ID de la persona
            location: Ubicación
            has_weapon: Si hay arma
            keypoints: Keypoints de la persona
            frame_image_path: Ruta a la imagen
        """
        alert_data = {
            'activity': activity,
            'risk_level': risk_level,
            'confidence': confidence,
            'person_id': person_id,
            'location': location,
            'has_weapon': has_weapon,
            'keypoints': keypoints,
            'frame_image_path': frame_image_path,
            'timestamp': datetime.now().isoformat()
        }
        
        # Agregar a lista local
        alert_id = len(self.pending_alerts) + 1
        local_alert = {
            'id': alert_id,
            **alert_data,
            'sent_at': datetime.now().isoformat()
        }
        self.pending_alerts.append(local_alert)
        
        # Enviar a app externa (Lovable) si está configurado
        if self.external_api_url and REQUESTS_AVAILABLE:
            try:
                self._send_to_external_api(alert_data)
            except Exception as e:
                print(f"⚠️  Error al enviar alerta a app externa: {e}")
        
        return alert_data
    
    def _transform_alert_for_supabase(self, alert_data: dict) -> dict:
        """
        Transforma los datos de alerta al formato que espera Supabase/Lovable
        
        Args:
            alert_data: Datos de alerta del sistema CV
            
        Returns:
            dict con formato para Supabase
        """
        # Mapear risk_level y activity a alert_type de Supabase
        activity = alert_data.get('activity', 'suspicious')
        risk_level = alert_data.get('risk_level', 'anómala')
        has_weapon = alert_data.get('has_weapon', False)
        
        # Determinar alert_type según actividad y riesgo
        if has_weapon or risk_level == 'delictiva':
            if activity == 'hurto':
                alert_type = 'theft'
            elif has_weapon:
                alert_type = 'weapon'
            else:
                alert_type = 'violence'
        else:
            alert_type = 'suspicious'
        
        # Extraer district de location si es posible
        location = alert_data.get('location', 'Desconocida')
        district = 'Santiago de Surco'  # Por defecto según el contexto del proyecto
        
        # Intentar extraer district de location si tiene formato específico
        if 'Surco' in location or 'surco' in location.lower():
            district = 'Santiago de Surco'
        elif 'Miraflores' in location or 'miraflores' in location.lower():
            district = 'Miraflores'
        
        # Generar camera_id si no existe
        camera_id = f"CAM-{district[:3].upper()}-{alert_data.get('person_id', 0)}"
        
        # Formatear timestamp en formato ISO 8601 con Z (UTC)
        timestamp = alert_data.get('timestamp')
        if timestamp:
            # Si ya es string ISO, asegurar que termine en Z
            if isinstance(timestamp, str):
                if not timestamp.endswith('Z') and '+' not in timestamp:
                    timestamp = timestamp.replace('+00:00', 'Z').replace('+00', 'Z')
                    if not timestamp.endswith('Z'):
                        timestamp += 'Z'
        else:
            # Generar timestamp actual en formato ISO con Z
            timestamp = datetime.utcnow().isoformat() + 'Z'
        
        # Transformar al formato Supabase
        supabase_alert = {
            'camera_id': camera_id,
            'alert_type': alert_type,
            'confidence_score': float(alert_data.get('confidence', 0.5)),
            'location': location,
            'district': district,
            'detected_at': timestamp,
            'video_frame_url': alert_data.get('frame_image_path'),
            'metadata': {
                'activity': activity,
                'risk_level': risk_level,
                'person_id': alert_data.get('person_id'),
                'keypoints': alert_data.get('keypoints'),
                'original_confidence': alert_data.get('confidence')
            }
        }
        
        return supabase_alert
    
    def _send_to_external_api(self, alert_data: dict):
        """
        Envía una alerta a la app externa (Lovable/Supabase) mediante HTTP POST
        
        Args:
            alert_data: Datos de la alerta a enviar
        """
        if not REQUESTS_AVAILABLE:
            print("⚠️  Requests no está disponible. Instala con: pip install requests")
            return
        
        if not self.external_api_url:
            return
        
        # Transformar datos al formato Supabase
        supabase_alert = self._transform_alert_for_supabase(alert_data)
        
        # Preparar headers
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        # Agregar API key si está configurado
        if self.external_api_key:
            headers['Authorization'] = f'Bearer {self.external_api_key}'
        
        # Preparar payload en formato Supabase
        payload = {
            'alert': supabase_alert
        }
        
        try:
            # Debug: mostrar payload que se envía
            import json
            print(f"\n📤 Enviando alerta a Supabase:")
            print(f"   URL: {self.external_api_url}")
            print(f"   Payload: {json.dumps(payload, indent=2, ensure_ascii=False)[:300]}...")
            
            # Enviar POST request
            response = requests.post(
                self.external_api_url,
                json=payload,
                headers=headers,
                timeout=10  # Timeout aumentado a 10 segundos
            )
            
            # Verificar respuesta
            print(f"   Status Code: {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            
            if response.status_code == 200 or response.status_code == 201:
                print(f"✅ Alerta enviada exitosamente a {self.external_api_url}")
                try:
                    response_data = response.json()
                    if 'alert_id' in response_data:
                        print(f"   Alert ID: {response_data.get('alert_id')}")
                except:
                    pass
            else:
                print(f"❌ Error al enviar alerta: Status {response.status_code}")
                print(f"   Respuesta completa: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"❌ Timeout al enviar alerta a {self.external_api_url}")
        except requests.exceptions.ConnectionError as e:
            print(f"❌ Error de conexión con {self.external_api_url}")
            print(f"   Detalles: {e}")
        except Exception as e:
            print(f"❌ Error al enviar alerta: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self, debug=False, threaded=True):
        """
        Ejecuta el servidor Flask
        
        Args:
            debug: Modo debug
            threaded: Ejecutar en thread separado
        """
        if not FLASK_AVAILABLE or self.app is None:
            print("⚠️  Flask no está disponible. La API REST no se iniciará.")
            return
        
        if threaded:
            thread = threading.Thread(
                target=self.app.run,
                kwargs={'host': self.host, 'port': self.port, 'debug': debug, 'use_reloader': False}
            )
            thread.daemon = True
            thread.start()
            print(f"✅ API REST iniciada en http://{self.host}:{self.port}")
        else:
            self.app.run(host=self.host, port=self.port, debug=debug)
    
    def stop(self):
        """Detiene el servidor Flask"""
        # Flask no tiene un método directo para detener, pero podemos usar un shutdown hook
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        func()

