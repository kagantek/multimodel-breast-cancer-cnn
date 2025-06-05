from flask import Flask, render_template, request

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/notebook')
def notebook():
    return render_template('notebook.html')

@app.route('/resnet_mammography_notebook')
def resnet_mammography_notebook():
    return render_template('resnet_mammography_notebook.html')

@app.route('/vgg_mammography_notebook')
def vgg_mammography_notebook():
    return render_template('vgg_mammography_notebook.html')

@app.route('/densenet_mammography_notebook')
def densenet_mammography_notebook():
    return render_template('densenet_mammography_notebook.html')

@app.route('/resnet_ultrasound_notebook')
def resnet_ultrasound_notebook():
    return render_template('resnet_ultrasound_notebook.html')

@app.route('/vgg_ultrasound_notebook')
def vgg_ultrasound_notebook():
    return render_template('vgg_ultrasound_notebook.html')

@app.route('/densenet_ultrasound_notebook')
def densenet_ultrasound_notebook():
    return render_template('densenet_ultrasound_notebook.html')

@app.route('/resnet_histopathology_notebook')
def resnet_histopathology_notebook():
    return render_template('resnet_histopathology_notebook.html')

@app.route('/vgg_histopathology_notebook')
def vgg_histopathology_notebook():
    return render_template('vgg_histopathology_notebook.html')

@app.route('/densenet_histopathology_notebook')
def densenet_histopathology_notebook():
    return render_template('densenet_histopathology_notebook.html')

@app.route('/resnet_mammography', methods=['GET', 'POST'])
def resnet_mammography():
    if request.method == 'POST':
        imagefile = request.files['imagefile']

        images_dir = "./static/images"
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        image_path = os.path.join(images_dir, imagefile.filename)
        imagefile.save(image_path)

        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        print("Loading ResNet50 Mammography Model...")
        modelResnet = load_model('./model/mammography/mammography_resnet.h5', compile=False, custom_objects={'preprocess_input': preprocess_input})
        print("Model loaded successfully!")

        yhat = modelResnet.predict(image, verbose=0)
        prob = float(yhat[0][0])
        TAU = 0.49
        label = "Tumor Detected" if prob >= TAU else "No Tumor Detected"
        conf = prob if prob >= TAU else 1 - prob
        prediction = f"{label} ({conf*100:.2f}%) (Threshold: {TAU})"

        return render_template('resnet_mammography.html', prediction=prediction)

    return render_template('resnet_mammography.html')


@app.route('/vgg_mammography')
def vgg_mammography():
    return render_template('vgg_mammography.html')

@app.route('/densenet_mammography')
def densenet_mammography():
    return render_template('densenet_mammography.html')

@app.route('/resnet_ultrasound')
def resnet_ultrasound():
    return render_template('resnet_ultrasound.html')

@app.route('/vgg_ultrasound')
def vgg_ultrasound():
    return render_template('vgg_ultrasound.html')

@app.route('/densenet_ultrasound')
def densenet_ultrasound():
    return render_template('densenet_ultrasound.html')

@app.route('/resnet_histopathology')
def resnet_histopathology():
    return render_template('resnet_histopathology.html')

@app.route('/vgg_histopathology')
def vgg_histopathology():
    return render_template('vgg_histopathology.html')

@app.route('/densenet_histopathology')
def densenet_histopathology():
    return render_template('densenet_histopathology.html') 

if __name__ == '__main__':
    app.run(debug=True)