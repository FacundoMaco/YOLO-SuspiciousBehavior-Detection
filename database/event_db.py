"""
Módulo de base de datos para almacenar eventos detectados
Usa SQLite para almacenamiento local
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
import os


class EventDatabase:
    """
    Gestor de base de datos para eventos de videovigilancia
    """
    
    def __init__(self, db_path='database/events.db'):
        """
        Inicializa la conexión a la base de datos
        
        Args:
            db_path: Ruta al archivo de base de datos
        """
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Para acceder a columnas por nombre
        self.create_tables()
    
    def create_tables(self):
        """Crea las tablas necesarias si no existen"""
        cursor = self.conn.cursor()
        
        # Tabla de eventos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                activity TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                confidence REAL NOT NULL,
                person_id INTEGER,
                location TEXT,
                has_weapon INTEGER DEFAULT 0,
                frame_image_path TEXT,
                keypoints TEXT,
                alert_sent INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabla de estadísticas del sistema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                total_events INTEGER DEFAULT 0,
                safe_events INTEGER DEFAULT 0,
                anomalous_events INTEGER DEFAULT 0,
                criminal_events INTEGER DEFAULT 0,
                false_positives INTEGER DEFAULT 0,
                response_time_avg REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    def insert_event(self, activity: str, risk_level: str, confidence: float,
                    person_id: Optional[int] = None, location: Optional[str] = None,
                    has_weapon: bool = False, frame_image_path: Optional[str] = None,
                    keypoints: Optional[List] = None, alert_sent: bool = False):
        """
        Inserta un nuevo evento en la base de datos
        
        Args:
            activity: Actividad detectada
            risk_level: Nivel de riesgo (segura, anómala, delictiva)
            confidence: Confianza de la detección
            person_id: ID de la persona (opcional)
            location: Ubicación del evento (opcional)
            has_weapon: Si se detectó un arma
            frame_image_path: Ruta a la imagen del frame (opcional)
            keypoints: Keypoints de la persona (opcional)
            alert_sent: Si se envió una alerta
            
        Returns:
            ID del evento insertado
        """
        cursor = self.conn.cursor()
        
        timestamp = datetime.now().isoformat()
        keypoints_json = json.dumps(keypoints) if keypoints else None
        
        cursor.execute('''
            INSERT INTO events 
            (timestamp, activity, risk_level, confidence, person_id, location,
             has_weapon, frame_image_path, keypoints, alert_sent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, activity, risk_level, confidence, person_id, location,
              int(has_weapon), frame_image_path, keypoints_json, int(alert_sent)))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_events(self, limit: int = 100, risk_level: Optional[str] = None,
                   start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        Obtiene eventos de la base de datos
        
        Args:
            limit: Número máximo de eventos a retornar
            risk_level: Filtrar por nivel de riesgo (opcional)
            start_date: Fecha de inicio (opcional)
            end_date: Fecha de fin (opcional)
            
        Returns:
            Lista de eventos
        """
        cursor = self.conn.cursor()
        
        query = 'SELECT * FROM events WHERE 1=1'
        params = []
        
        if risk_level:
            query += ' AND risk_level = ?'
            params.append(risk_level)
        
        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND timestamp <= ?'
            params.append(end_date)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Convertir a lista de diccionarios
        events = []
        for row in rows:
            event = dict(row)
            # Parsear keypoints si existen
            if event['keypoints']:
                try:
                    event['keypoints'] = json.loads(event['keypoints'])
                except:
                    event['keypoints'] = None
            events.append(event)
        
        return events
    
    def get_statistics(self, days: int = 7):
        """
        Obtiene estadísticas del sistema
        
        Args:
            days: Número de días hacia atrás para las estadísticas
            
        Returns:
            dict con estadísticas
        """
        cursor = self.conn.cursor()
        
        # Contar eventos por nivel de riesgo
        cursor.execute('''
            SELECT risk_level, COUNT(*) as count
            FROM events
            WHERE timestamp >= datetime('now', '-' || ? || ' days')
            GROUP BY risk_level
        ''', (days,))
        
        risk_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Estadísticas generales
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                AVG(confidence) as avg_confidence,
                COUNT(CASE WHEN alert_sent = 1 THEN 1 END) as alerts_sent
            FROM events
            WHERE timestamp >= datetime('now', '-' || ? || ' days')
        ''', (days,))
        
        stats_row = cursor.fetchone()
        
        return {
            'total_events': stats_row[0] if stats_row else 0,
            'safe_events': risk_counts.get('segura', 0),
            'anomalous_events': risk_counts.get('anómala', 0),
            'criminal_events': risk_counts.get('delictiva', 0),
            'avg_confidence': stats_row[1] if stats_row and stats_row[1] else 0.0,
            'alerts_sent': stats_row[2] if stats_row else 0,
            'days': days
        }
    
    def mark_alert_sent(self, event_id: int):
        """
        Marca un evento como que se envió una alerta
        
        Args:
            event_id: ID del evento
        """
        cursor = self.conn.cursor()
        cursor.execute('UPDATE events SET alert_sent = 1 WHERE id = ?', (event_id,))
        self.conn.commit()
    
    def close(self):
        """Cierra la conexión a la base de datos"""
        self.conn.close()

