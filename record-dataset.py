import os
import cv2
import time
import numpy as np
import mediapipe as mp

# ────────────────────────────────────────────────
# 🗂️ Crear carpetas para almacenar imágenes por clase
# ────────────────────────────────────────────────
output_dirs = ["dataset/piedra", "dataset/papel", "dataset/tijeras"]
for dir in output_dirs:
    os.makedirs(dir, exist_ok=True)  # Crea las carpetas si no existen

# ────────────────────────────────────────────────
# ✋ Inicializar MediaPipe para detección de manos
# ────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)  # modo video, una sola mano
mp_draw = mp.solutions.drawing_utils  # Para dibujar los landmarks en la imagen

# 🧠 Inicializar listas para almacenar coordenadas y etiquetas
data = []
labels = []

# 🎥 Captura de video desde la webcam
cap = cv2.VideoCapture(0)

print("Presiona 0 para piedra, 1 para papel, 2 para tijeras. ESC para salir.")

# ────────────────────────────────────────────────
# 🔁 Bucle principal para capturar gestos
# ────────────────────────────────────────────────
while True:
    ret, frame = cap.read()  # Leer un frame de la cámara
    if not ret:
        break  # Si no se puede leer, salir del bucle

    # Convertir el frame a RGB (MediaPipe lo requiere)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)  # Procesar con MediaPipe

    # Si se detecta una mano en el frame
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            coords = []  # Lista para almacenar coordenadas de la mano

            # Extraer coordenadas x e y de los 21 puntos
            for lm in hand_landmarks.landmark:
                coords.append(lm.x)
                coords.append(lm.y)

            # Mostrar instrucciones en el frame
            instrucciones = "Presiona 0 (Piedra), 1 (Papel), 2 (Tijeras), ESC (salir)"
            cv2.putText(frame, instrucciones, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Dibujar los puntos de la mano detectada
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Leer tecla presionada
            key = cv2.waitKey(1)

            # Si se presiona una tecla válida (0, 1, 2)
            if key in [ord('0'), ord('1'), ord('2')]:
                label = int(chr(key))  # Convertir de ASCII a int
                data.append(coords)   # Guardar coordenadas
                labels.append(label)  # Guardar etiqueta (clase)

                # Determinar el nombre de la clase según el valor
                label_name = ["piedra", "papel", "tijeras"][label]
                
                # Guardar imagen con nombre único basado en el timestamp
                filename = f"{int(time.time() * 1000)}.jpg"
                save_path = os.path.join("dataset", label_name, filename)
                cv2.imwrite(save_path, frame)

                # Mostrar en consola
                gesto = ["Piedra", "Papel", "Tijeras"][label]
                print(f"[✓] Gesto guardado: {gesto} ({label}) - Imagen: {save_path}")

                # Mostrar mensaje en pantalla por 1 segundo
                cv2.putText(frame, f"Grabado: {gesto}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
                cv2.imshow("Grabando dataset", frame)
                cv2.waitKey(1000)  # Pausar 1 segundo para que el usuario vea el mensaje

    # Mostrar el frame en pantalla continuamente
    cv2.imshow("Grabando dataset", frame)

    # Si se presiona ESC (código ASCII 27), salir
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 🔚 Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()

# 💾 Guardar coordenadas y etiquetas
# np.save("rps_dataset.npy", np.array(data))
# np.save("rps_labels.npy", np.array(labels))
