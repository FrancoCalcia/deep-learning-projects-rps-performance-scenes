# ✋ Rock, Paper, Scissors - Clasificador de gestos con MediaPipe + Keras

Este proyecto permite capturar gestos desde la cámara web, entrenar un modelo de clasificación y luego predecir en tiempo real los gestos de **Piedra**, **Papel** y **Tijeras**.

Se usa **MediaPipe** para detectar los puntos clave de la mano, y una **red neuronal entrenada con Keras** para clasificar el gesto.

---

## 📁 Estructura del proyecto

```
rock-paper-scissors/
├── capturas/                     # Carpeta para imágenes guardadas durante la predicción
├── dataset.rar                   # 📦 Dataset comprimido (ver nota abajo)
├── record-dataset.py             # Script para grabar gestos y construir el dataset
├── train-gesture-classifier.py   # Script para entrenar la red neuronal
├── rock-paper-scissors.py        # Script para predecir el gesto en tiempo real
├── rps_dataset.npy               # Coordenadas x,y de cada gesto
├── rps_labels.npy                # Etiquetas correspondientes a los gestos
├── rps_model.h5                  # Modelo entrenado guardado
├── rps_scaler_mean.npy           # Media del escalador
├── rps_scaler_scale.npy          # Escala del escalador
├── requirements.txt              # Dependencias necesarias
└── README.md                     # Este archivo
```

---

## ⚙️ Requisitos

Instalación de dependencias necesarias:

```bash
pip install -r requirements.txt
```

Contenido típico del `requirements.txt`:

```
opencv-python
mediapipe
tensorflow
numpy
scikit-learn
matplotlib
jupyter
```

---

## 🚀 Instrucciones de uso

### 1. Recolectar datos

Ejecutá el siguiente script:

```bash
python record-dataset.py
```

Presioná:
- `0` para Piedra
- `1` para Papel
- `2` para Tijeras
- `ESC` para salir

> Las imágenes se guardan en la carpeta `dataset/` según la clase y se genera el archivo `.npy` con los datos.

---

### 2. Entrenar el modelo

```bash
python train-gesture-classifier.py
```

Este script:
- Escala los datos
- Codifica las etiquetas en formato one-hot
- Entrena una red neuronal
- Guarda el modelo y el escalador

---

### 3. Clasificación en tiempo real

```bash
python rock-paper-scissors.py
```

- Muestra la predicción del gesto en la pantalla.
- Si presionás la **barra espaciadora**, se guarda una captura en la carpeta `capturas/`.
- Presioná **ESC** para salir.

---

## 📦 Sobre el dataset

La carpeta `dataset/` fue comprimida como `dataset.rar` para poder subirla a GitHub, ya que contiene muchas imágenes.  
Descomprimí el archivo antes de entrenar el modelo para que `train-gesture-classifier.py` funcione correctamente.

---

## ⚠️ Nota técnica

Durante el desarrollo del proyecto hubo varios problemas relacionados con la instalación de librerías, especialmente con **MediaPipe** y **TensorFlow**.  
Después de probar diferentes configuraciones, se resolvieron los conflictos utilizando **Python 3.11.9**, que fue compatible con todas las dependencias sin errores de instalación ni ejecución.

> 💡 Recomendación: si encontrás errores al instalar las librerías, probá crear un entorno virtual con Python 3.11.9.

