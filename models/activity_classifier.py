"""
Modelo CNN para clasificación de actividades humanas
Basado en el artículo de Sathiyavathi et al. (2021)
Clasifica: caminar, sentarse, interactuar, saludar, hurto
"""

import numpy as np
import os

# TensorFlow es opcional - solo se necesita si hay modelo entrenado
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    keras = None
    layers = None


class ActivityClassifier:
    """
    Clasificador de actividades usando CNN
    Entrada: keypoints de pose (17 puntos × 2 coordenadas = 34 features)
    Salida: Probabilidades para 5 actividades
    """
    
    ACTIVITIES = ['caminar', 'sentarse', 'interactuar', 'saludar', 'hurto']
    NUM_ACTIVITIES = len(ACTIVITIES)
    
    def __init__(self, model_path=None, input_shape=(34,)):
        """
        Inicializa el clasificador de actividades
        
        Args:
            model_path: Ruta al modelo pre-entrenado (opcional)
            input_shape: Forma de entrada (34 features de keypoints)
        """
        self.model_path = model_path
        self.input_shape = input_shape
        self.model = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()
    
    def build_model(self):
        """
        Construye el modelo CNN para clasificación de actividades
        Arquitectura basada en el artículo: CNN con capas convolucionales
        """
        if not TENSORFLOW_AVAILABLE:
            # Si TensorFlow no está disponible, no construir modelo
            # Se usará clasificación basada en reglas
            self.model = None
            return None
        
        model = keras.Sequential([
            # Capa de entrada: 34 features (17 keypoints × 2 coordenadas)
            layers.Input(shape=self.input_shape),
            
            # Normalización
            layers.BatchNormalization(),
            
            # Primera capa densa con dropout
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            
            # Segunda capa densa
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            # Tercera capa densa
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Capa de salida: 5 actividades
            layers.Dense(self.NUM_ACTIVITIES, activation='softmax')
        ])
        
        # Compilar modelo
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def preprocess_keypoints(self, keypoints):
        """
        Preprocesa los keypoints para la entrada del modelo
        
        Args:
            keypoints: Array de keypoints (17 puntos × 2 coordenadas)
            
        Returns:
            Array normalizado y aplanado
        """
        if keypoints is None or len(keypoints) == 0:
            return np.zeros(self.input_shape)
        
        # Convertir a numpy array si no lo es
        keypoints = np.array(keypoints)
        
        # Si tiene forma (17, 2), aplanar a (34,)
        if keypoints.shape == (17, 2):
            keypoints = keypoints.flatten()
        elif keypoints.shape == (17, 3):
            # Si incluye confianza, tomar solo x, y
            keypoints = keypoints[:, :2].flatten()
        
        # Asegurar que tenga la forma correcta
        if len(keypoints) != 34:
            # Rellenar con ceros si faltan puntos
            padded = np.zeros(34)
            padded[:min(len(keypoints), 34)] = keypoints[:34]
            keypoints = padded
        
        # Normalizar coordenadas (normalización simple)
        # En producción, usar normalización basada en estadísticas del dataset
        if np.max(np.abs(keypoints)) > 0:
            keypoints = keypoints / (np.max(np.abs(keypoints)) + 1e-8)
        
        return keypoints.reshape(1, -1)  # Añadir dimensión de batch
    
    def predict_with_rules(self, keypoints):
        """Clasificación simple basada en reglas de pose"""
        if keypoints is None or len(keypoints) == 0:
            return {'activity': 'caminar', 'confidence': 0.5}
        
        kpts = np.array(keypoints)
        if kpts.shape == (17, 3):
            kpts = kpts[:, :2]
        elif len(kpts) == 34:
            kpts = kpts.reshape(17, 2)
        elif kpts.shape != (17, 2):
            return {'activity': 'caminar', 'confidence': 0.5}
        
        # Filtrar puntos válidos
        valid = (kpts[:, 0] > 0) & (kpts[:, 1] > 0)
        if np.sum(valid) < 5:
            return {'activity': 'caminar', 'confidence': 0.5}
        
        scores = {'caminar': 0.3, 'sentarse': 0.0, 'interactuar': 0.0, 'saludar': 0.0, 'hurto': 0.0}
        
        # SENTARSE: cadera baja
        if valid[11] and valid[12] and valid[13] and valid[14]:
            hip_y = (kpts[11, 1] + kpts[12, 1]) / 2
            knee_y = (kpts[13, 1] + kpts[14, 1]) / 2
            if valid[0]:  # nariz
                if hip_y > kpts[0, 1] * 0.7:  # cadera baja
                    scores['sentarse'] = 0.8
        
        # SALUDAR: brazo levantado
        if valid[5] and valid[9] and kpts[9, 1] < kpts[5, 1]:
            scores['saludar'] = 0.7
        if valid[6] and valid[10] and kpts[10, 1] < kpts[6, 1]:
            scores['saludar'] = 0.7
        
        # INTERACTUAR: brazos extendidos
        if valid[5] and valid[6] and valid[9] and valid[10]:
            arm_ext = abs(kpts[9, 0] - kpts[5, 0]) + abs(kpts[10, 0] - kpts[6, 0])
            if arm_ext > 100:
                scores['interactuar'] = 0.6
        
        # CAMINAR: piernas separadas
        if valid[15] and valid[16]:
            leg_sep = abs(kpts[15, 0] - kpts[16, 0])
            if leg_sep > 30:
                scores['caminar'] = 0.7
        
        # Normalizar
        total = sum(scores.values())
        if total > 0:
            for k in scores:
                scores[k] /= total
        
        activity = max(scores, key=scores.get)
        return {'activity': activity, 'confidence': scores[activity]}
    
    def predict(self, keypoints):
        """
        Predice la actividad basada en los keypoints
        Usa reglas si no hay modelo entrenado, CNN si hay modelo
        
        Args:
            keypoints: Array de keypoints de pose
            
        Returns:
            dict con actividad predicha y probabilidades
        """
        # Si no hay modelo entrenado, usar reglas basadas en pose
        if self.model is None:
            return self.predict_with_rules(keypoints)
        
        # Si hay modelo entrenado, usar CNN
        # Preprocesar keypoints
        processed = self.preprocess_keypoints(keypoints)
        
        # Predecir
        predictions = self.model.predict(processed, verbose=0)[0]
        
        # Obtener actividad con mayor probabilidad
        activity_idx = np.argmax(predictions)
        activity = self.ACTIVITIES[activity_idx]
        confidence = float(predictions[activity_idx])
        
        # Crear diccionario de probabilidades
        probabilities = {
            self.ACTIVITIES[i]: float(pred) 
            for i, pred in enumerate(predictions)
        }
        
        return {
            'activity': activity,
            'confidence': confidence,
            'probabilities': probabilities
        }
    
    def predict_batch(self, keypoints_list):
        """
        Predice actividades para múltiples personas
        
        Args:
            keypoints_list: Lista de arrays de keypoints
            
        Returns:
            Lista de diccionarios con predicciones
        """
        results = []
        for keypoints in keypoints_list:
            results.append(self.predict(keypoints))
        return results
    
    def save_model(self, path):
        """
        Guarda el modelo entrenado
        
        Args:
            path: Ruta donde guardar el modelo
        """
        if not TENSORFLOW_AVAILABLE:
            print("⚠️  TensorFlow no está disponible, no se puede guardar modelo")
            return
        
        if self.model is None:
            return
        
        self.model.save(path)
        print(f"✅ Modelo guardado en: {path}")
    
    def load_model(self, path):
        """
        Carga un modelo pre-entrenado
        
        Args:
            path: Ruta al modelo
        """
        if not TENSORFLOW_AVAILABLE:
            print(f"⚠️  TensorFlow no está disponible, no se puede cargar modelo")
            print("   Usando clasificación basada en reglas")
            self.model = None
            return
        
        try:
            self.model = keras.models.load_model(path)
            print(f"✅ Modelo cargado desde: {path}")
        except Exception as e:
            print(f"❌ Error al cargar modelo: {e}")
            print("   Construyendo modelo nuevo...")
            self.build_model()
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=20, batch_size=32):
        """
        Entrena el modelo (para uso futuro con dataset)
        
        Args:
            X_train: Datos de entrenamiento (keypoints)
            y_train: Etiquetas de entrenamiento (one-hot encoded)
            X_val: Datos de validación (opcional)
            y_val: Etiquetas de validación (opcional)
            epochs: Número de épocas
            batch_size: Tamaño del batch
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow no está disponible. Instala con: pip install tensorflow")
        
        if self.model is None:
            self.build_model()
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                'models/best_activity_model.h5',
                save_best_only=True,
                monitor='val_loss' if X_val is not None else 'loss'
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history

