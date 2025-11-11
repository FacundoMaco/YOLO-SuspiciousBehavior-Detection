"""
Análisis temporal simplificado - MVP
"""

from collections import deque


class TemporalAnalyzer:
    """Analizador temporal simplificado"""
    
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.sequences = {}
    
    def update_sequence(self, person_id, activity, confidence):
        """Actualiza secuencia de una persona"""
        if person_id not in self.sequences:
            self.sequences[person_id] = deque(maxlen=self.sequence_length)
        self.sequences[person_id].append(activity)
    
    def analyze_sequence(self, person_id):
        """Análisis simple - retorna actividad más común"""
        if person_id not in self.sequences or len(self.sequences[person_id]) < 3:
            return {'is_anomalous': False, 'pattern': 'normal'}
        
        seq = list(self.sequences[person_id])
        most_common = max(set(seq), key=seq.count)
        is_anomalous = most_common == 'hurto' or len(set(seq)) > 3
        
        return {
            'is_anomalous': is_anomalous,
            'pattern': 'criminal' if most_common == 'hurto' else 'normal',
            'most_common_activity': most_common
        }


