"""
Análisis temporal de secuencias de video
Detecta cambios inusuales en comportamientos mediante análisis de secuencias
"""

import numpy as np
from collections import deque
from typing import List, Dict, Optional


class TemporalAnalyzer:
    """
    Analiza secuencias temporales de actividades para detectar comportamientos anómalos
    """
    
    def __init__(self, sequence_length=30, anomaly_threshold=0.7):
        """
        Inicializa el analizador temporal
        
        Args:
            sequence_length: Longitud de la secuencia a analizar (número de frames)
            anomaly_threshold: Umbral para considerar una secuencia como anómala
        """
        self.sequence_length = sequence_length
        self.anomaly_threshold = anomaly_threshold
        
        # Buffer para almacenar secuencias de cada persona
        # Estructura: {person_id: deque([activity1, activity2, ...])}
        self.activity_sequences = {}
        
        # Historial de actividades por persona
        self.activity_history = {}
    
    def update_sequence(self, person_id, activity, confidence):
        """
        Actualiza la secuencia de actividades para una persona
        
        Args:
            person_id: ID único de la persona
            activity: Actividad detectada
            confidence: Confianza de la detección
        """
        if person_id not in self.activity_sequences:
            self.activity_sequences[person_id] = deque(maxlen=self.sequence_length)
            self.activity_history[person_id] = []
        
        # Agregar actividad a la secuencia
        self.activity_sequences[person_id].append({
            'activity': activity,
            'confidence': confidence
        })
        
        # Mantener historial completo
        self.activity_history[person_id].append({
            'activity': activity,
            'confidence': confidence
        })
    
    def analyze_sequence(self, person_id):
        """
        Analiza la secuencia de actividades de una persona para detectar anomalías
        
        Args:
            person_id: ID de la persona a analizar
            
        Returns:
            dict con análisis de la secuencia
        """
        if person_id not in self.activity_sequences:
            return {
                'is_anomalous': False,
                'anomaly_score': 0.0,
                'pattern': 'no_data',
                'most_common_activity': None
            }
        
        sequence = list(self.activity_sequences[person_id])
        
        if len(sequence) < 5:  # Necesitamos al menos 5 frames
            return {
                'is_anomalous': False,
                'anomaly_score': 0.0,
                'pattern': 'insufficient_data',
                'most_common_activity': sequence[-1]['activity'] if sequence else None
            }
        
        # Contar frecuencia de actividades
        activity_counts = {}
        for item in sequence:
            activity = item['activity']
            activity_counts[activity] = activity_counts.get(activity, 0) + 1
        
        # Actividad más común
        most_common_activity = max(activity_counts, key=activity_counts.get)
        most_common_count = activity_counts[most_common_activity]
        
        # Calcular score de anomalía basado en:
        # 1. Variabilidad de actividades (más variabilidad = más anómalo)
        # 2. Presencia de actividades delictivas
        # 3. Cambios bruscos en patrones
        
        num_unique_activities = len(activity_counts)
        total_frames = len(sequence)
        
        # Score de variabilidad (0-1)
        variability_score = min(num_unique_activities / 5.0, 1.0)
        
        # Score por presencia de actividades delictivas
        criminal_score = 0.0
        if 'hurto' in activity_counts:
            criminal_score = activity_counts['hurto'] / total_frames
        
        # Score por cambios bruscos (detectar transiciones rápidas)
        transition_score = 0.0
        if len(sequence) > 1:
            transitions = 0
            for i in range(1, len(sequence)):
                if sequence[i]['activity'] != sequence[i-1]['activity']:
                    transitions += 1
            transition_score = min(transitions / (len(sequence) - 1), 1.0)
        
        # Score combinado de anomalía
        anomaly_score = (
            variability_score * 0.3 +
            criminal_score * 0.5 +
            transition_score * 0.2
        )
        
        is_anomalous = anomaly_score >= self.anomaly_threshold
        
        # Determinar patrón
        if criminal_score > 0.3:
            pattern = 'criminal'
        elif variability_score > 0.6:
            pattern = 'high_variability'
        elif transition_score > 0.5:
            pattern = 'unstable'
        else:
            pattern = 'normal'
        
        return {
            'is_anomalous': is_anomalous,
            'anomaly_score': anomaly_score,
            'pattern': pattern,
            'most_common_activity': most_common_activity,
            'activity_distribution': activity_counts,
            'variability_score': variability_score,
            'criminal_score': criminal_score,
            'transition_score': transition_score
        }
    
    def detect_unusual_movement(self, person_id, current_keypoints, previous_keypoints):
        """
        Detecta movimientos inusuales comparando keypoints entre frames
        
        Args:
            person_id: ID de la persona
            current_keypoints: Keypoints del frame actual
            previous_keypoints: Keypoints del frame anterior
            
        Returns:
            dict con información sobre el movimiento
        """
        if previous_keypoints is None or current_keypoints is None:
            return {
                'is_unusual': False,
                'movement_magnitude': 0.0
            }
        
        # Calcular desplazamiento promedio de los keypoints
        if len(current_keypoints) != len(previous_keypoints):
            return {
                'is_unusual': False,
                'movement_magnitude': 0.0
            }
        
        # Calcular distancia euclidiana entre keypoints correspondientes
        displacements = []
        for i in range(min(len(current_keypoints), len(previous_keypoints))):
            if len(current_keypoints[i]) >= 2 and len(previous_keypoints[i]) >= 2:
                dx = current_keypoints[i][0] - previous_keypoints[i][0]
                dy = current_keypoints[i][1] - previous_keypoints[i][1]
                distance = np.sqrt(dx**2 + dy**2)
                displacements.append(distance)
        
        if not displacements:
            return {
                'is_unusual': False,
                'movement_magnitude': 0.0
            }
        
        movement_magnitude = np.mean(displacements)
        
        # Considerar movimiento inusual si es muy rápido o muy lento
        # (ajustar umbrales según necesidad)
        is_unusual = movement_magnitude > 50 or movement_magnitude < 2
        
        return {
            'is_unusual': is_unusual,
            'movement_magnitude': movement_magnitude
        }
    
    def remove_person(self, person_id):
        """
        Elimina los datos de una persona del analizador
        
        Args:
            person_id: ID de la persona a eliminar
        """
        if person_id in self.activity_sequences:
            del self.activity_sequences[person_id]
        if person_id in self.activity_history:
            del self.activity_history[person_id]
    
    def get_person_statistics(self, person_id):
        """
        Obtiene estadísticas de una persona
        
        Args:
            person_id: ID de la persona
            
        Returns:
            dict con estadísticas
        """
        if person_id not in self.activity_history:
            return None
        
        history = self.activity_history[person_id]
        
        if not history:
            return None
        
        activity_counts = {}
        for item in history:
            activity = item['activity']
            activity_counts[activity] = activity_counts.get(activity, 0) + 1
        
        return {
            'total_frames': len(history),
            'activity_distribution': activity_counts,
            'most_common_activity': max(activity_counts, key=activity_counts.get) if activity_counts else None
        }

