"""
Sistema de clasificación de riesgo basado en actividades detectadas
Clasifica en 3 niveles: segura, anómala, delictiva
"""

from models.activity_classifier import ActivityClassifier


class RiskClassifier:
    """
    Clasificador de riesgo basado en actividades y contexto
    Niveles: segura, anómala, delictiva
    """
    
    RISK_LEVELS = ['segura', 'anómala', 'delictiva']
    
    # Actividades normales (seguras)
    SAFE_ACTIVITIES = ['caminar', 'sentarse', 'saludar']
    
    # Actividades anómalas (requieren atención)
    ANOMALOUS_ACTIVITIES = ['interactuar']
    
    # Actividades delictivas
    CRIMINAL_ACTIVITIES = ['hurto']
    
    def __init__(self):
        """Inicializa el clasificador de riesgo"""
        pass
    
    def classify_risk(self, activity, has_weapon=False, activity_confidence=0.0):
        """
        Clasifica el nivel de riesgo basado en la actividad detectada
        
        Args:
            activity: Actividad detectada (caminar, sentarse, etc.)
            has_weapon: Si se detectó un arma o cuchillo
            activity_confidence: Confianza de la detección de actividad
            
        Returns:
            dict con nivel de riesgo y justificación
        """
        # Si hay arma, siempre es delictiva
        if has_weapon:
            return {
                'risk_level': 'delictiva',
                'confidence': 1.0,
                'reason': 'Presencia de arma detectada',
                'activity': activity
            }
        
        # Si la actividad es hurto, es delictiva
        if activity in self.CRIMINAL_ACTIVITIES:
            return {
                'risk_level': 'delictiva',
                'confidence': activity_confidence,
                'reason': f'Actividad delictiva detectada: {activity}',
                'activity': activity
            }
        
        # Si la actividad es anómala, clasificar como anómala
        if activity in self.ANOMALOUS_ACTIVITIES:
            # Si la confianza es alta, es más probable que sea anómala
            if activity_confidence > 0.7:
                return {
                    'risk_level': 'anómala',
                    'confidence': activity_confidence,
                    'reason': f'Actividad anómala detectada: {activity}',
                    'activity': activity
                }
            else:
                # Si la confianza es baja, podría ser falsa alarma
                return {
                    'risk_level': 'segura',
                    'confidence': 1.0 - activity_confidence,
                    'reason': f'Actividad normal con baja confianza en detección anómala',
                    'activity': activity
                }
        
        # Actividades seguras
        if activity in self.SAFE_ACTIVITIES:
            return {
                'risk_level': 'segura',
                'confidence': activity_confidence if activity_confidence > 0 else 0.8,
                'reason': f'Actividad normal: {activity}',
                'activity': activity
            }
        
        # Actividad desconocida - clasificar como anómala por precaución
        return {
            'risk_level': 'anómala',
            'confidence': 0.5,
            'reason': f'Actividad desconocida o no reconocida: {activity}',
            'activity': activity
        }
    
    def get_risk_color(self, risk_level):
        """
        Obtiene el color para visualización según el nivel de riesgo
        
        Args:
            risk_level: Nivel de riesgo (segura, anómala, delictiva)
            
        Returns:
            Tupla BGR (Blue, Green, Red) para OpenCV
        """
        color_map = {
            'segura': (0, 255, 0),      # Verde
            'anómala': (0, 165, 255),   # Naranja
            'delictiva': (0, 0, 255)    # Rojo
        }
        return color_map.get(risk_level, (255, 255, 255))  # Blanco por defecto
    
    def get_risk_label(self, risk_level):
        """
        Obtiene la etiqueta en español para el nivel de riesgo
        
        Args:
            risk_level: Nivel de riesgo
            
        Returns:
            String con la etiqueta
        """
        label_map = {
            'segura': 'SEGURA',
            'anómala': 'ANÓMALA',
            'delictiva': 'DELICTIVA'
        }
        return label_map.get(risk_level, 'DESCONOCIDA')

