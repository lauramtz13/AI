import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file

app = Flask(__name__)

# Limitar la memoria GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])  # Limitar a 1 GB
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Obtener los archivos seleccionados por el usuario
        content_file = request.files['content']
        style_file = request.files['style']

        # Guardar los archivos en el directorio actual
        content_path = './content.jpg'
        style_path = './style.jpg'
        content_file.save(content_path)
        style_file.save(style_path)

        # Verificar si la imagen de contenido es de formato JPG
        if os.path.splitext(content_path)[1].lower() == '.jpg':
            # Cargar la imagen de contenido
            content = Image.open(content_path)
        else:
            print("La imagen de contenido no es de formato JPG.")
            content = None

        # Verificar si la imagen de estilo es de formato PNG o JPG
        if os.path.splitext(style_path)[1].lower() in ['.png', '.jpg', '.jpeg']:
            # Cargar la imagen de estilo
            style = Image.open(style_path)
        else:
            print("La imagen de estilo no es de formato PNG o JPG.")
            style = None

        # Cargar el módulo de estilo arbitrario
        hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
        hub_module = hub.load(hub_handle)

        # Cargar las imágenes de contenido y estilo
        content_image = np.array(content)[np.newaxis, ...].astype(np.float32) / 255.0
        style_image = np.array(style)[np.newaxis, ...].astype(np.float32) / 255.0

        # Estilizar la imagen de contenido usando la imagen de estilo
        outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
        stylized_image = np.squeeze(outputs[0])  # Eliminar las dimensiones adicionales

        # Guardar la imagen generada en un archivo temporal sin redimensionar
        output_path = './output.jpg'
        output_image = Image.fromarray((stylized_image * 255).astype(np.uint8))
        output_image.save(output_path)

        return send_file(output_path, as_attachment=True)

    return render_template('index.html')


if __name__ == '__main__':
    app.run()
