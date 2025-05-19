🤖 **Deep Learning Projects - Clasificación y Regresión con Keras + MediaPipe**

Este repositorio contiene tres proyectos basados en redes neuronales, enfocados en:

* Clasificación de gestos con la mano (Piedra, Papel o Tijeras) _-> Ejercicio 2_
* Predicción del rendimiento académico _-> Ejercicio 1_
* Clasificación de imágenes de escenas naturales _-> Ejercicio 3_

Se utilizaron herramientas como TensorFlow/Keras, MediaPipe, scikit-learn, OpenCV, entre otras.

---

📚 **CONTENIDO DEL REPOSITORIO**

1. ✋ Rock, Paper, Scissors - Clasificador de Gestos
2. 📘 Predicción del Rendimiento Académico
3. 🌄 Clasificación de Imágenes de Escenas Naturales

---

✋ **1. Rock, Paper, Scissors - Clasificador de Gestos**

Este proyecto permite capturar gestos de la mano a través de la webcam, entrenar un modelo de red neuronal y luego clasificar en tiempo real los gestos de Piedra, Papel o Tijeras.

Se utiliza MediaPipe para extraer los puntos clave (landmarks) de la mano, y Keras para entrenar un modelo de clasificación.

🛠️ **Requisitos**
Instalación de dependencias necesarias:

```bash
pip install -r requirements.txt
```

Contenido típico del archivo requirements.txt:

* opencv-python
* mediapipe
* tensorflow
* numpy
* scikit-learn
* matplotlib
* jupyter

🚀 **Instrucciones de uso**

1. **Recolección de datos**
   Ejecutar:
   `python record-dataset.py`
   Presionar:

   * 0 para Piedra
   * 1 para Papel
   * 2 para Tijeras
   * ESC para salir

2. **Entrenamiento del modelo**
   Ejecutar:
   `python train-gesture-classifier.py`
   El script:

   * Escala los datos
   * Codifica las etiquetas en formato one-hot
   * Entrena la red neuronal
   * Guarda el modelo entrenado y el escalador

3. **Clasificación en tiempo real**
   Ejecutar:
   `python rock-paper-scissors.py`

   * Muestra la predicción en pantalla
   * Presionar barra espaciadora para guardar una captura
   * Presionar ESC para salir

🗂️ **Sobre el dataset**

El conjunto de imágenes se encuentra comprimido como `dataset.rar`. Descomprimir antes de ejecutar el entrenamiento.

⚠️ **Nota técnica**

Para evitar errores de compatibilidad con MediaPipe y TensorFlow, se recomienda usar **Python 3.11.9** en un entorno virtual.

---

📘 **2. Predicción del Rendimiento Académico**

Este proyecto implementa un modelo de regresión para predecir el índice de rendimiento (`Performance Index`) de estudiantes universitarios, utilizando una red neuronal densa.

**Variables del dataset:**

* Horas de estudio
* Puntuaciones previas
* Participación en actividades extracurriculares (Sí/No)
* Horas promedio de sueño
* Cantidad de cuestionarios practicados

**Requisitos:**

* pandas
* matplotlib
* seaborn
* scikit-learn
* tensorflow

📊 **Instrucciones de uso:**

1. Subir el archivo `academic_data.csv` o montar Google Drive desde Colab.
2. Ejecutar el notebook (ejercicio 1) `AA2-TP1-Avecilla-Calcia-Rizzotto.ipynb`.
3. El flujo incluye:

   * Análisis exploratorio de datos
   * Preprocesamiento (inclusión de variables categóricas)
   * Normalización
   * Entrenamiento de red neuronal
   * Evaluación mediante MAE, RMSE y R²
   * Visualización de predicciones reales vs. estimadas

---

🌄 **3. Clasificación de Imágenes de Escenas Naturales**

Este proyecto entrena y compara distintos modelos de clasificación de imágenes en seis categorías:

* buildings
* forest
* glacier
* mountain
* sea
* street

**Estructura del módulo:**

scene-classification/

* train/ → Carpeta con imágenes de entrenamiento
* test/ → Carpeta con imágenes de test
* prediction/ → Carpeta con imágenes sin etiqueta
* cnn\_models.ipynb → Notebook con los modelos implementados
* models/ → Carpeta opcional para guardar modelos entrenados

🧱 **Modelos implementados:**

1. Red neuronal densa básica
2. Red neuronal convolucional tradicional
3. Red con bloque residual simple (ResNet-like)
4. Transfer Learning usando MobileNetV2 como backbone

⚙️ **Instrucciones de uso:**

1. Descomprimir el dataset si es necesario.
2. Ejecutar el notebook (ejercicio 3) `AA2-TP1-Avecilla-Calcia-Rizzotto.ipynb`.
3. El flujo incluye:

   * Carga y preprocesamiento de imágenes (150x150 px)
   * Entrenamiento de los diferentes modelos
   * Evaluación con métricas de clasificación (accuracy, matriz de confusión)
   * Visualización de ejemplos predichos
   * Comparación de rendimiento entre arquitecturas

**Notas adicionales:**

* Las imágenes deben tener tamaño 150x150.
* Se utiliza `ImageDataGenerator` para cargar datos y aplicar aumentos.


