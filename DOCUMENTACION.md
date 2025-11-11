# Smart surveillance system for abnormal activity detection using CNN

**Integrantes:**
- Maria Fernanda Tapia Yepez
- Marianet Leon Astuhuaman
- Mariana Emy Sanchez Galdos
- Manuel Aarón Torres Tolentino

## Introducción

El artículo desarrollado por Sathiyavathi, Jessey, Selvakumar y SaiRamesh (2021) aborda un problema fundamental en el ámbito de la seguridad tecnológica y urbana: la incapacidad de los sistemas tradicionales de videovigilancia para anticipar comportamientos delictivos. Los autores señalan que la mayoría de los sistemas actuales registran los hechos solo después de que estos ocurren, lo cual limita la capacidad de prevención y genera una alta cantidad de falsas alarmas. Esta deficiencia impide la detección temprana de actividades sospechosas y la activación oportuna de mecanismos de respuesta. Por ello, el estudio parte de la necesidad de diseñar una herramienta inteligente capaz de analizar comportamientos humanos en tiempo real y emitir alertas preventivas antes de que se produzca un incidente, utilizando inteligencia artificial y procesamiento de imágenes.

## Aporte Principal del Estudio

El principal aporte del estudio es el desarrollo de un sistema de videovigilancia inteligente que utiliza redes neuronales convolucionales (CNN) y el algoritmo OpenPose para analizar imágenes captadas por cámaras y reconocer patrones de movimiento humano. A diferencia de los sistemas tradicionales que reaccionan después de ocurrido un incidente, este modelo permite identificar y clasificar actividades como seguras, anormales o delictivas en tiempo real, enviando alertas automáticas al usuario o administrador. Además, contribuye al avance del campo de la seguridad automatizada, ya que reduce la intervención humana, disminuye los falsos positivos y aumenta la precisión en la detección de amenazas. Su aplicación puede adaptarse tanto a entornos públicos como privados, mejorando la capacidad de respuesta ante incidentes y fortaleciendo las estrategias de prevención del delito (Sathiyavathi et al., 2021).

El principal aporte del artículo consiste en la creación de un sistema de videovigilancia inteligente que integra redes neuronales convolucionales (CNN) con el algoritmo OpenPose para identificar y clasificar comportamientos humanos. Este modelo logra distinguir entre actividades seguras, anómalas o delictivas, generando alertas automáticas que se envían al usuario o administrador (Sathiyavathi et al., 2021). A diferencia de los sistemas convencionales, la propuesta se basa en el uso de aprendizaje profundo para transformar la videovigilancia pasiva en un sistema preventivo, reduciendo la intervención humana y aumentando la precisión del monitoreo. Este enfoque representa una contribución relevante en el campo de la seguridad automatizada, al ofrecer una estructura adaptable a diferentes contextos urbanos y a las necesidades de las instituciones que buscan fortalecer su capacidad de reacción ante delitos.

## Resultados del Estudio

Los resultados cuantitativos obtenidos por los autores evidencian la eficacia de la propuesta. En la etapa de entrenamiento, el sistema utilizó 2 950 videos distribuidos en cinco tipos de actividad y 329 videos adicionales para validación, desarrollados a lo largo de 20 épocas de aprendizaje. Gracias a este proceso, el modelo fue capaz de reconocer con alta precisión acciones humanas como caminar, sentarse, interactuar, saludar o cometer un hurto. Asimismo, el sistema mostró en pantalla los puntos corporales detectados, el número de personas presentes y la actividad correspondiente en cada fotograma, logrando una clasificación precisa en tiempo real. Los investigadores concluyen que la integración del algoritmo OpenPose con CNN reduce significativamente las falsas alarmas y mejora la detección oportuna de incidentes, confirmando la validez del modelo en entornos urbanos complejos (Sathiyavathi et al., 2021).

## Contexto y Relevancia

En el contexto del Objetivo de Desarrollo Sostenible 16, que promueve la paz, la justicia y el fortalecimiento de las instituciones, el problema abordado resulta especialmente relevante. En el distrito de Santiago de Surco, los casos de robo aumentaron un 102,89 % entre 2021 y 2024 (Espinoza, 2024), lo que refleja una creciente inseguridad ciudadana y la necesidad de incorporar tecnologías de videovigilancia que permitan una respuesta preventiva y eficiente. Bajo esta perspectiva, se plantea la siguiente propuesta de aplicación teórica del modelo desarrollado por Sathiyavathi et al. (2021), orientada a la innovación en sistemas de seguridad urbana inteligentes.

## Propuesta de Aplicación Teórica

### Fase 1: Adquisición y procesamiento de datos visuales

* Integrar cámaras de videovigilancia municipales y privadas mediante una red de transmisión en tiempo real, garantizando conectividad segura.

* Recopilar imágenes de zonas críticas del distrito de Surco para generar una base de datos inicial representativa de comportamientos cotidianos y sospechosos.

* Procesar los fotogramas capturados con redes neuronales convolucionales (CNN) entrenadas en TensorFlow y Keras para identificar patrones de movimiento humano.

* Implementar un sistema capaz de reconocer cambios inusuales en la conducta o desplazamientos atípicos mediante análisis de secuencias de video.

### Fase 2: Estimación de la postura humana

* Utilizar el algoritmo OpenPose para extraer puntos corporales clave (articulaciones y extremidades) y generar mapas de calor (heatmaps) y campos de afinidad (PAFs).

* Analizar la posición y la relación entre los puntos del cuerpo para reconstruir posturas humanas y detectar comportamientos potencialmente violentos o irregulares.

* Comparar las posturas registradas con modelos previamente entrenados para distinguir entre actividades seguras y sospechosas, como forcejeos o movimientos bruscos.

### Fase 3: Clasificación conductual y gestión de alertas

* Incorporar un modelo de aprendizaje supervisado que clasifique la actividad en tres niveles de riesgo: segura, anómala o delictiva.

* Emitir notificaciones automáticas al centro de control del municipio con la imagen, la hora y la ubicación exacta del incidente.

* Integrar la comunicación a través de plataformas API REST o sistemas de mensajería en tiempo real para garantizar la acción inmediata de las autoridades competentes.

### Fase 4: Retroalimentación y aprendizaje adaptativo

* Registrar los eventos detectados en una base de datos dinámica que permita evaluar la precisión del sistema y su tasa de aciertos.

* Utilizar los datos recolectados para reentrenar el modelo y mejorar continuamente su capacidad predictiva mediante aprendizaje autónomo.

* Visualizar indicadores de desempeño (alertas correctas, falsas alarmas, tiempos de respuesta) para retroalimentar las políticas municipales de seguridad.

## Estructura del Sistema

La implementación de estas fases permitiría organizar la vigilancia inteligente de manera estructurada, combinando niveles institucionales, tecnológicos y operativos. En el nivel de conducción, se consolidaría la infraestructura tecnológica y el soporte institucional para integrar las cámaras existentes. En el nivel de enlace, se ubican los procesos de análisis visual y clasificación automatizada mediante CNN y OpenPose. Finalmente, en el nivel de resultados dependientes, se lograrían efectos concretos como la reducción de tiempos de respuesta ante emergencias, la detección temprana de delitos y el fortalecimiento de la confianza ciudadana.

En conjunto, esta adaptación del modelo al contexto urbano de Surco permitiría avanzar hacia una gestión de seguridad más proactiva, transparente y sostenible, contribuyendo directamente al cumplimiento del ODS 16.

## Características del Sistema

### Componentes Principales

1. **Redes Neuronales Convolucionales (CNN)**
   - Entrenadas con TensorFlow y Keras
   - Procesamiento de imágenes en tiempo real
   - Identificación de patrones de movimiento humano

2. **Algoritmo OpenPose**
   - Extracción de puntos corporales clave
   - Generación de mapas de calor (heatmaps)
   - Campos de afinidad (PAFs)
   - Reconstrucción de posturas humanas

3. **Clasificación de Actividades**
   - Reconocimiento de acciones: caminar, sentarse, interactuar, saludar, hurto
   - Clasificación en tres niveles: segura, anómala, delictiva
   - Análisis de secuencias temporales de video

4. **Sistema de Alertas**
   - Notificaciones automáticas en tiempo real
   - Integración con sistemas externos mediante API REST
   - Registro de eventos con información detallada (imagen, hora, ubicación)

5. **Base de Datos y Retroalimentación**
   - Almacenamiento de eventos detectados
   - Evaluación de precisión del sistema
   - Reentrenamiento continuo del modelo
   - Visualización de indicadores de desempeño

## Arquitectura Técnica

### Entrenamiento del Modelo

- **Dataset de entrenamiento**: 2,950 videos distribuidos en cinco tipos de actividad
- **Dataset de validación**: 329 videos adicionales
- **Épocas de entrenamiento**: 20 épocas
- **Actividades reconocidas**: caminar, sentarse, interactuar, saludar, hurto

### Procesamiento en Tiempo Real

- Captura de video desde cámaras de videovigilancia
- Procesamiento frame por frame con CNN
- Extracción de pose con OpenPose
- Clasificación de actividad en tiempo real
- Generación de alertas automáticas

### Salidas del Sistema

- Puntos corporales detectados visualizados en pantalla
- Número de personas presentes en cada fotograma
- Actividad clasificada para cada persona
- Nivel de riesgo asignado (segura, anómala, delictiva)
- Alertas automáticas con información contextual

## Referencias

* Espinoza, A. (2024, 16 de agosto). Miraflores y Surco en la cima del ranking de delincuencia en Lima: distritos exclusivos registran incremento de robos. Infobae. https://www.infobae.com/peru/2024/08/16/miraflores-y-surco-en-la-cima-del-ranking-de-delincuencia-en-lima-distritos-exclusivos-registran-incremento-de-robos/

* Sathiyavathi, V., Jessey, M., Selvakumar, K., & SaiRamesh, L. (2021). Smart surveillance system for abnormal activity detection using CNN. In D. J. Hemanth (Ed.), Advances in Parallel Computing Technologies and Applications (pp. 341–349). Scopus. https://www.scopus.com/pages/publications/85119889063?origin=resultslist

