from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np

import os

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return (
        '.' in filename and 
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )

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

@app.route('/mammography_notebook')
def resnet_mammography_notebook():
    return render_template('mammography_notebook.html')

@app.route('/ultrasound_notebook')
def vgg_mammography_notebook():
    return render_template('ultrasound_notebook.html')

@app.route('/densenet_notebook')
def densenet_mammography_notebook():
    return render_template('densenet_notebook.html')


@app.route('/resnet_mammography', methods=['GET', 'POST'])
def resnet_mammography():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from tensorflow.keras.models import load_model
    
    model_path = './model/mammography/mammography_resnet.h5'
    if not hasattr(resnet_mammography, 'model'):
        print("Loading ResNet50 Mammography Model...")
        resnet_mammography.model = load_model(
            model_path,
            compile=False,
            custom_objects={'preprocess_input': preprocess_input}
        )
        print("Model loaded successfully!")

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            error = "No file part in the request."
            return render_template('resnet_mammography.html', prediction=error)

        file = request.files['imagefile']

        if file.filename == '':
            error = "No file selected."
            return render_template('resnet_mammography.html', prediction=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Only .png, .jpg and .jpeg are allowed."
            return render_template('resnet_mammography.html', prediction=error)

        filename = secure_filename(file.filename)
        images_dir = "./static/images"
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, filename)
        file.save(image_path)

        try:
            img = load_img(image_path, target_size=(224, 224))
            x   = img_to_array(img)
            x   = np.expand_dims(x, axis=0)
            x   = preprocess_input(x)

            yhat = resnet_mammography.model.predict(x, verbose=0)
            prob = float(yhat[0][0])
            TAU  = 0.39
            if prob >= TAU:
                label = "Tumor Detected"
                conf  = prob
            else:
                label = "No Tumor Detected"
                conf  = 1 - prob

            prediction = f"{label} ({conf*100:.2f}%)  Threshold: {TAU}"

        finally:
            if os.path.exists(image_path):
                os.remove(image_path)

        return render_template('resnet_mammography.html', prediction=prediction)

    return render_template('resnet_mammography.html')


@app.route('/vgg_mammography', methods=['GET', 'POST'])
def vgg_mammography():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    from tensorflow.keras.models import load_model
    
    model_path = './model/mammography/mammography_vgg.h5'
    if not hasattr(vgg_mammography, 'model'):
        print("Loading VGG16 Mammography Model...")
        resnet_mammography.model = load_model(
            model_path,
            compile=False,
            custom_objects={'preprocess_input': preprocess_input}
        )
        print("Model loaded successfully!")

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            error = "No file part in the request."
            return render_template('vgg_mammography.html', prediction=error)

        file = request.files['imagefile']

        if file.filename == '':
            error = "No file selected."
            return render_template('vgg_mammography.html', prediction=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Only .png, .jpg and .jpeg are allowed."
            return render_template('vgg_mammography.html', prediction=error)

        filename = secure_filename(file.filename)
        images_dir = "./static/images"
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, filename)
        file.save(image_path)

        try:
            img = load_img(image_path, target_size=(224, 224))
            x   = img_to_array(img)
            x   = np.expand_dims(x, axis=0)
            x   = preprocess_input(x)

            yhat = vgg_mammography.model.predict(x, verbose=0)
            prob = float(yhat[0][0])
            TAU  = 0.49
            if prob >= TAU:
                label = "Tumor Detected"
                conf  = prob
            else:
                label = "No Tumor Detected"
                conf  = 1 - prob

            prediction = f"{label} ({conf*100:.2f}%)  Threshold: {TAU}"

        finally:
            if os.path.exists(image_path):
                os.remove(image_path)

        return render_template('vgg_mammography.html', prediction=prediction)

    return render_template('vgg_mammography.html')


@app.route('/densenet_mammography', methods=['GET', 'POST'])
def densenet_mammography():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
    from tensorflow.keras.models import load_model

    model_path = './model/mammography/mammography_densenet.h5'
    if not hasattr(densenet_mammography, 'model'):
        print("Loading DenseNet121 Mammography Model...")
        densenet_mammography.model = load_model(
            model_path,
            compile=False,
            custom_objects={'preprocess_input': preprocess_input}
        )
        print("Model loaded successfully!")

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            error = "No file part in the request."
            return render_template('densenet_mammography.html', prediction=error)

        file = request.files['imagefile']

        if file.filename == '':
            error = "No file selected."
            return render_template('densenet_mammography.html', prediction=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Only .png, .jpg and .jpeg are allowed."
            return render_template('densenet_mammography.html', prediction=error)

        filename = secure_filename(file.filename)
        images_dir = "./static/images"
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, filename)
        file.save(image_path)

        try:
            img = load_img(image_path, target_size=(224, 224))
            x   = img_to_array(img)
            x   = np.expand_dims(x, axis=0)
            x   = preprocess_input(x)

            yhat = densenet_mammography.model.predict(x, verbose=0)
            prob = float(yhat[0][0])
            TAU  = 0.22
            if prob >= TAU:
                label = "Tumor Detected"
                conf  = prob
            else:
                label = "No Tumor Detected"
                conf  = 1 - prob

            prediction = f"{label} ({conf*100:.2f}%)  Threshold: {TAU}"

        finally:
            if os.path.exists(image_path):
                os.remove(image_path)

        return render_template('densenet_mammography.html', prediction=prediction)

    return render_template('densenet_mammography.html')


@app.route('/resnet_ultrasound', methods=['GET', 'POST'])
def resnet_ultrasound():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from tensorflow.keras.models import load_model
    
    model_path = './model/ultrasound/ultrasound_resnet.h5'
    if not hasattr(resnet_ultrasound, 'model'):
        print("Loading ResNet50 Ultrasound Model...")
        resnet_ultrasound.model = load_model(
            model_path,
            compile=False,
            custom_objects={'preprocess_input': preprocess_input}
        )
        print("Model loaded successfully!")

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            error = "No file part in the request."
            return render_template('resnet_ultrasound.html', prediction=error)

        file = request.files['imagefile']

        if file.filename == '':
            error = "No file selected."
            return render_template('resnet_ultrasound.html', prediction=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Only .png, .jpg and .jpeg are allowed."
            return render_template('resnet_ultrasound.html', prediction=error)

        filename = secure_filename(file.filename)
        images_dir = "./static/images"
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, filename)
        file.save(image_path)

        try:
            img = load_img(image_path, target_size=(224, 224))
            x   = img_to_array(img)
            x   = np.expand_dims(x, axis=0)
            x   = preprocess_input(x)

            yhat = resnet_ultrasound.model.predict(x, verbose=0)
            prob = float(yhat[0][0])
            TAU  = 0.49
            if prob >= TAU:
                label = "Tumor Detected"
                conf  = prob
            else:
                label = "No Tumor Detected"
                conf  = 1 - prob

            prediction = f"{label} ({conf*100:.2f}%)  Threshold: {TAU}"

        finally:
            if os.path.exists(image_path):
                os.remove(image_path)

        return render_template('resnet_ultrasound.html', prediction=prediction)

    return render_template('resnet_ultrasound.html')

@app.route('/vgg_ultrasound', methods=['GET', 'POST'])
def vgg_ultrasound():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    from tensorflow.keras.models import load_model

    model_path = './model/ultrasound/ultrasound_vgg16.h5'
    if not hasattr(vgg_ultrasound, 'model'):
        print("Loading VGG16 Ultrasound Model...")
        vgg_ultrasound.model = load_model(
            model_path,
            compile=False,
            custom_objects={'preprocess_input': preprocess_input}
        )
        print("Model loaded successfully!")

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            error = "No file part in the request."
            return render_template('vgg_ultrasound.html', prediction=error)

        file = request.files['imagefile']

        if file.filename == '':
            error = "No file selected."
            return render_template('vgg_ultrasound.html', prediction=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Only .png, .jpg and .jpeg are allowed."
            return render_template('vgg_ultrasound.html', prediction=error)

        filename = secure_filename(file.filename)
        images_dir = "./static/images"
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, filename)
        file.save(image_path)

        try:
            img = load_img(image_path, target_size=(224, 224))
            x   = img_to_array(img)
            x   = np.expand_dims(x, axis=0)
            x   = preprocess_input(x)

            yhat = vgg_ultrasound.model.predict(x, verbose=0)
            prob = float(yhat[0][0])
            TAU  = 0.49
            if prob >= TAU:
                label = "Tumor Detected"
                conf  = prob
            else:
                label = "No Tumor Detected"
                conf  = 1 - prob

            prediction = f"{label} ({conf*100:.2f}%)  Threshold: {TAU}"

        finally:
            if os.path.exists(image_path):
                os.remove(image_path)

        return render_template('vgg_ultrasound.html', prediction=prediction)

    return render_template('vgg_ultrasound.html')

@app.route('/densenet_ultrasound', methods=['GET', 'POST'])
def densenet_ultrasound():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
    from tensorflow.keras.models import load_model

    model_path = './model/ultrasound/ultrasound_densenet.h5'
    if not hasattr(densenet_ultrasound, 'model'):
        print("Loading DenseNet121 Ultrasound Model...")
        densenet_ultrasound.model = load_model(
            model_path,
            compile=False,
            custom_objects={'preprocess_input': preprocess_input}
        )
        print("Model loaded successfully!")

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            error = "No file part in the request."
            return render_template('densenet_ultrasound.html', prediction=error)

        file = request.files['imagefile']

        if file.filename == '':
            error = "No file selected."
            return render_template('densenet_ultrasound.html', prediction=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Only .png, .jpg and .jpeg are allowed."
            return render_template('densenet_ultrasound.html', prediction=error)

        filename = secure_filename(file.filename)
        images_dir = "./static/images"
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, filename)
        file.save(image_path)

        try:
            img = load_img(image_path, target_size=(224, 224))
            x   = img_to_array(img)
            x   = np.expand_dims(x, axis=0)
            x   = preprocess_input(x)

            yhat = densenet_ultrasound.model.predict(x, verbose=0)
            prob = float(yhat[0][0])
            TAU  = 0.22
            if prob >= TAU:
                label = "Tumor Detected"
                conf  = prob
            else:
                label = "No Tumor Detected"
                conf  = 1 - prob

            prediction = f"{label} ({conf*100:.2f}%)  Threshold: {TAU}"

        finally:
            if os.path.exists(image_path):
                os.remove(image_path)

        return render_template('densenet_ultrasound.html', prediction=prediction)

    return render_template('densenet_ultrasound.html')

@app.route('/resnet_histopathology', methods=['GET', 'POST'])
def resnet_histopathology():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from tensorflow.keras.models import load_model

    model_path = './model/histopathology/resnet_histopathology.h5'
    if not hasattr(resnet_histopathology, 'model'):
        print("Loading ResNet50 Histopathology Model...")
        resnet_histopathology.model = load_model(
            model_path,
            compile=False,
            custom_objects={'preprocess_input': preprocess_input}
        )
        print("Model loaded successfully!")

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            error = "No file part in the request."
            return render_template('resnet_histopathology.html', prediction=error)

        file = request.files['imagefile']

        if file.filename == '':
            error = "No file selected."
            return render_template('resnet_histopathology.html', prediction=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Only .png, .jpg and .jpeg are allowed."
            return render_template('resnet_histopathology.html', prediction=error)

        filename = secure_filename(file.filename)
        images_dir = "./static/images"
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, filename)
        file.save(image_path)

        try:
            img = load_img(image_path, target_size=(224, 224))
            x   = img_to_array(img)
            x   = np.expand_dims(x, axis=0)
            x   = preprocess_input(x)

            yhat = resnet_histopathology.model.predict(x, verbose=0)
            prob = float(yhat[0][0])
            TAU  = 0.46
            if prob >= TAU:
                label = "Tumor Detected"
                conf  = prob
            else:
                label = "No Tumor Detected"
                conf  = 1 - prob

            prediction = f"{label} ({conf*100:.2f}%)  Threshold: {TAU}"

        finally:
            if os.path.exists(image_path):
                os.remove(image_path)

        return render_template('resnet_histopathology.html', prediction=prediction)

    return render_template('resnet_histopathology.html')

@app.route('/vgg_histopathology', methods=['GET', 'POST'])
def vgg_histopathology():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    from tensorflow.keras.models import load_model

    model_path = './model/histopathology/vgg_histopathology.h5'
    if not hasattr(vgg_histopathology, 'model'):
        print("Loading VGG16 Histopathology Model...")
        vgg_histopathology.model = load_model(
            model_path,
            compile=False,
            custom_objects={'preprocess_input': preprocess_input}
        )
        print("Model loaded successfully!")

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            error = "No file part in the request."
            return render_template('vgg_histopathology.html', prediction=error)

        file = request.files['imagefile']

        if file.filename == '':
            error = "No file selected."
            return render_template('vgg_histopathology.html', prediction=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Only .png, .jpg and .jpeg are allowed."
            return render_template('vgg_histopathology.html', prediction=error)

        filename = secure_filename(file.filename)
        images_dir = "./static/images"
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, filename)
        file.save(image_path)

        try:
            img = load_img(image_path, target_size=(224, 224))
            x   = img_to_array(img)
            x   = np.expand_dims(x, axis=0)
            x   = preprocess_input(x)

            yhat = vgg_histopathology.model.predict(x, verbose=0)
            prob = float(yhat[0][0])
            TAU  = 0.50
            if prob >= TAU:
                label = "Tumor Detected"
                conf  = prob
            else:
                label = "No Tumor Detected"
                conf  = 1 - prob

            prediction = f"{label} ({conf*100:.2f}%)  Threshold: {TAU}"

        finally:
            if os.path.exists(image_path):
                os.remove(image_path)

        return render_template('vgg_histopathology.html', prediction=prediction)

    return render_template('vgg_histopathology.html')

@app.route('/densenet_histopathology', methods=['GET', 'POST'])
def densenet_histopathology():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
    from tensorflow.keras.models import load_model

    model_path = './model/histopathology/densenet_histopathology.h5'
    if not hasattr(densenet_histopathology, 'model'):
        print("Loading DenseNet121 Histopathology Model...")
        densenet_histopathology.model = load_model(
            model_path,
            compile=False,
            custom_objects={'preprocess_input': preprocess_input}
        )
        print("Model loaded successfully!")

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            error = "No file part in the request."
            return render_template('densenet_histopathology.html', prediction=error)

        file = request.files['imagefile']

        if file.filename == '':
            error = "No file selected."
            return render_template('densenet_histopathology.html', prediction=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Only .png, .jpg and .jpeg are allowed."
            return render_template('densenet_histopathology.html', prediction=error)

        filename = secure_filename(file.filename)
        images_dir = "./static/images"
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, filename)
        file.save(image_path)

        try:
            img = load_img(image_path, target_size=(224, 224))
            x   = img_to_array(img)
            x   = np.expand_dims(x, axis=0)
            x   = preprocess_input(x)

            yhat = densenet_histopathology.model.predict(x, verbose=0)
            prob = float(yhat[0][0])
            TAU  = 0.46
            if prob >= TAU:
                label = "Tumor Detected"
                conf  = prob
            else:
                label = "No Tumor Detected"
                conf  = 1 - prob

            prediction = f"{label} ({conf*100:.2f}%)  Threshold: {TAU}"

        finally:
            if os.path.exists(image_path):
                os.remove(image_path)

        return render_template('densenet_histopathology.html', prediction=prediction)

    return render_template('densenet_histopathology.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=True)