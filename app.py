import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Cargar el modelo entrenado
modelo = tf.keras.models.load_model("clasificacion-perros-gatos-CNNAD.h5")  # Asegúrate de tener tu modelo en la misma carpeta

# Función para procesar la imagen y hacer una predicción
def predecir(imagen):
    if not isinstance(imagen, np.ndarray):
        imagen = np.array(imagen)  # Convierte de tensor o PIL a NumPy

    imagen = cv2.resize(imagen, (100, 100))

    # Convertir a escala de grises si el modelo espera imágenes en 1 canal
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Normalizar la imagen (0 a 1)
    imagen = imagen / 255.0  

    # Agregar dimensión de canal si el modelo espera imágenes en escala de grises
    imagen = np.expand_dims(imagen, axis=-1)  

    # Agregar dimensión batch (1, 100, 100, 1)
    imagen = np.expand_dims(imagen, axis=0)  

      # Hacer la predicción
    prediccion = modelo.predict(imagen)


    # Suponiendo que la salida es una probabilidad entre 0 y 1 (0 = gato, 1 = perro)
    clase = "Perro 🐶" if prediccion[0][0] > 0.5 else "Gato 🐱"
    confianza = prediccion[0][0] if prediccion[0][0] > 0.5 else 1 - prediccion[0][0]
    return clase, confianza

# Interfaz en Streamlit
st.title("Clasificador de Perros y Gatos 🐶🐱")
st.write("Sube una imagen o usa la cámara para predecir si es un perro o un gato.")

# Opción 1: Subir imagen
imagen_subida = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

if imagen_subida:
    imagen = Image.open(imagen_subida)
    ##st.image(imagen, caption="Imagen cargada", use_column_width=True)
    st.image(imagen,caption="Imagen cargada", use_container_width=True)
    # Hacer la predicción
    clase, confianza = predecir(imagen)
    
    # Mostrar resultado
    st.write(f"**Predicción:** {clase}")
    st.write(f"**Confianza:** {confianza:.2%}")

# Opción 2: Usar la cámara
st.write("---")
st.write("O usa la cámara para capturar una imagen:")

captura = st.camera_input("Toma una foto")

if captura:
    imagen = Image.open(captura)
    st.image(imagen, caption="Imagen capturada", use_column_width=True)

    # Hacer la predicción
    clase, confianza = predecir(imagen)

    # Mostrar resultado
    st.write(f"**Predicción:** {clase}")
    st.write(f"**Confianza:** {confianza:.2%}")
