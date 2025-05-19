import os
import cv2
import time
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# 1. Cargar el modelo entrenado y los parámetros del escalador
model = load_model("rps_model.h5")
scaler_mean = np.load("rps_scaler_mean.npy")
scaler_scale = np.load("rps_scaler_scale.npy")

# Función para escalar la entrada como se hizo durante el entrenamiento
def scale_input(X):
    return (X - scaler_mean) / scaler_scale

# 2. Inicializar MediaPipe Hands para detectar la mano en tiempo real
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)  # modo video, 2 manos
mp_draw = mp.solutions.drawing_utils  # para dibujar los landmarks

# 3. Lista de clases mapeadas según su índice
clases = ["Piedra", "Papel", "Tijeras"]

# 4. Iniciar la captura de video desde la webcam
cap = cv2.VideoCapture(0)

print("🙌 Mostrá un gesto y el sistema lo va a clasificar en tiempo real...\nPara guardar una imagen presiona la barra espaciadora")

# Bucle principal que corre hasta que se presione ESC
while True:
    ret, frame = cap.read()
    if not ret:
        break  # si no se puede capturar el frame, salir

    # Convertir el frame de BGR a RGB (MediaPipe requiere RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen con MediaPipe para detectar la mano
    result = hands.process(frame_rgb)

    # Si se detecta al menos una mano
    if result.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            coords = []
            for lm in hand_landmarks.landmark:
                coords.append(lm.x)
                coords.append(lm.y)

            if len(coords) == 42:
                X_input = scale_input(np.array(coords).reshape(1, -1))
                prediction = model.predict(X_input, verbose=0)
                class_id = np.argmax(prediction)
                prob = prediction[0][class_id]
                label = f"{clases[class_id]} ({prob*100:.1f}%)"

                # Dibujar la mano
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Obtener posición base para ubicar el texto sobre cada mano
                x = int(hand_landmarks.landmark[0].x * frame.shape[1])
                y = int(hand_landmarks.landmark[0].y * frame.shape[0]) - 20

                # Dibujar la etiqueta en la imagen cerca de la mano
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)


    # Mostrar el frame en una ventana
    cv2.imshow("Rock Paper Scissors", frame)

    # Leer tecla presionada
    key = cv2.waitKey(1) & 0xFF

    # Si se presiona la barra espaciadora (ASCII 32), guardar el frame actual como imagen
    if key == 32:
        timestamp = int(time.time() * 1000)  # timestamp para nombre único
        filename = os.path.join("capturas", f"captura_{timestamp}.jpg")
        os.makedirs("capturas", exist_ok=True)  # crear carpeta si no existe
        cv2.imwrite(filename, frame)  # guardar la imagen
        print(f"[📸] Imagen guardada como {filename}")

    # Si se presiona ESC (ASCII 27), salir del bucle
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
