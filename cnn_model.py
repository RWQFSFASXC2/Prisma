import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks  # Importamos callbacks
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# 1. CARGA DE DATOS (Igual que el anterior)
def preparar_dataset():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["hackathon_db"]
    coleccion = db["dataset_entrenamiento"]

    print("⏳ Cargando imágenes desde MongoDB...")
    documentos = list(coleccion.find({"categoria_id": {"$ne": None}}))
    X, y = [], []
    for doc in documentos:
        img = cv2.imread(doc["ruta_fisica"])
        if img is None: continue
        img = cv2.resize(img, (128, 128))
        X.append(img.astype('float32') / 255.0)
        y.append(doc["categoria_id"])
    return train_test_split(np.array(X), np.array(y), test_size=0.2, random_state=42)


# 2. MODELO CNN
def crear_modelo():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# --- EJECUCIÓN CON EARLY STOPPING ---
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preparar_dataset()
    ia_modelo = crear_modelo()

    # 3. CONFIGURACIÓN DE EARLY STOPPING
    # monitor='val_loss': Vigila el error en los datos que la IA NO conoce.
    # patience=5: Si el error no mejora en 5 épocas seguidas, se detiene.
    # restore_best_weights=True: Al detenerse, vuelve a los pesos donde el error fue mínimo.
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    print("\n🚀 Entrenando con Early Stopping activo...")
    # Ahora podemos poner 50 épocas con confianza, se detendrá mucho antes si es necesario
    history = ia_modelo.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]  # <--- AGREGAMOS EL CALLBACK AQUÍ
    )

    # 4. GRÁFICAS Y MÉTRICAS
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['val_loss'], label='Error Validación (Val Loss)')
    plt.plot(history.history['loss'], label='Error Entrenamiento (Loss)')
    plt.title('Detección de Overfitting (Loss)')
    plt.legend()
    plt.show()

    print("\n✅ IA entrenada. El Early Stopping evitó el sobreentrenamiento.")
    ia_modelo.save('modelo_optimizado.h5')