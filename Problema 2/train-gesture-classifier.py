import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical


# 1. Cargar los datos
X = np.load("rps_dataset.npy")  # Shape: (n_samples, 42)
y = np.load("rps_labels.npy")   # Shape: (n_samples,)

# 2. Escalar los datos (opcional pero recomendado)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Codificar etiquetas (one-hot) para clasificación con softmax
y_cat = to_categorical(y, num_classes=3)

# 4. Dividir en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_cat, test_size=0.2, random_state=42)

# 5. Crear la red neuronal
model = Sequential([
    Input(shape=(42,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 3 clases: piedra, papel, tijeras
])

# 6. Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 7. Entrenar el modelo
history = model.fit(X_train, y_train, epochs=300, batch_size=16, validation_split=0.2)

# 8. Evaluar el modelo
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Precisión en test: {acc*100:.2f}%")

# 9. Guardar el modelo entrenado y el scaler
#model.save("rps_model.h5")
#np.save("rps_scaler_mean.npy", scaler.mean_)
#np.save("rps_scaler_scale.npy", scaler.scale_)

print("🧠 Modelo y scaler guardados correctamente.")
