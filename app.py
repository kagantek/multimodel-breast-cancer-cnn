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

@app.route('/comparison')
def comparison():
    return render_template('comparison.html')

@app.route('/mammography')
def mammography():
    return render_template('mammography.html')

@app.route('/ultrasound')
def ultrasound():
    return render_template('ultrasound.html')

@app.route('/histopathology')
def histopathology():
    return render_template('histopathology.html')

@app.route('/mammography_notebook')
def mammography_notebook():
    return render_template('mammography_notebook.html')

@app.route('/ultrasound_notebook')
def ultrasound_notebook():
    return render_template('ultrasound_notebook.html')

@app.route('/histopathology_notebook')
def histopathology_notebook():
    return render_template('histopathology_notebook.html')

@app.route('/resnet_mammography_initial_unfrozen', methods=['GET', 'POST'])
def resnet_mammography_initial_unfrozen():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from tensorflow.keras.models import load_model
    
    model_path = './model/mammography/initial_unfrozen/mammography_resnet.h5'
    if not hasattr(resnet_mammography_initial_unfrozen, 'model'):
        print("Loading ResNet50 Mammography Model...")
        resnet_mammography_initial_unfrozen.model = load_model(
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

            yhat = resnet_mammography_initial_unfrozen.model.predict(x, verbose=0)
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

        return render_template('resnet_mammography_initial_unfrozen.html', prediction=prediction)

    return render_template('resnet_mammography_initial_unfrozen.html')

@app.route('/resnet_mammography_finetuned', methods=['GET', 'POST'])
def resnet_mammography_finetuned():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from tensorflow.keras.models import load_model
    
    model_path = './model/mammography/fine_tune/mammography_resnet.h5'
    if not hasattr(resnet_mammography_finetuned, 'model'):
        print("Loading ResNet50 Mammography Model...")
        resnet_mammography_finetuned.model = load_model(
            model_path,
            compile=False,
            custom_objects={'preprocess_input': preprocess_input}
        )
        print("Model loaded successfully!")

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            error = "No file part in the request."
            return render_template('resnet_mammography_finetuned.html', prediction=error)

        file = request.files['imagefile']

        if file.filename == '':
            error = "No file selected."
            return render_template('resnet_mammography_finetuned.html', prediction=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Only .png, .jpg and .jpeg are allowed."
            return render_template('resnet_mammography_finetuned.html', prediction=error)

        filename = secure_filename(file.filename)
        images_dir = "./static/images"
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, filename)
        file.save(image_path)

        try:
            img = load_img(image_path, target_size=(224, 224))
            x   = img_to_array(img)
            x   = np.expand_dims(x, axis=0)

            yhat = resnet_mammography_finetuned.model.predict(x, verbose=0)
            prob = float(yhat[0][0])
            TAU  = 0.65
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

        return render_template('resnet_mammography_finetuned.html', prediction=prediction)

    return render_template('resnet_mammography_finetuned.html')


@app.route('/vgg_mammography_initial_unfrozen', methods=['GET', 'POST'])
def vgg_mammography_initial_unfrozen():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    from tensorflow.keras.models import load_model

    model_path = './model/mammography/initial_unfrozen/mammography_vgg.h5'
    if not hasattr(vgg_mammography_initial_unfrozen, 'model'):
        print("Loading VGG16 Mammography Model...")
        vgg_mammography_initial_unfrozen.model = load_model(
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

            yhat = vgg_mammography_initial_unfrozen.model.predict(x, verbose=0)
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

        return render_template('vgg_mammography_initial_unfrozen.html', prediction=prediction)

    return render_template('vgg_mammography_initial_unfrozen.html')

@app.route('/vgg_mammography_finetuned', methods=['GET', 'POST'])
def vgg_mammography_finetuned():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    from tensorflow.keras.models import load_model

    model_path = './model/mammography/fine_tune/mammography_vgg.h5'
    if not hasattr(vgg_mammography_finetuned, 'model'):
        print("Loading VGG16 Mammography Model...")
        vgg_mammography_finetuned.model = load_model(
            model_path,
            compile=False,
            custom_objects={'preprocess_input': preprocess_input}
        )
        print("Model loaded successfully!")

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            error = "No file part in the request."
            return render_template('vgg_mammography_finetuned.html', prediction=error)

        file = request.files['imagefile']

        if file.filename == '':
            error = "No file selected."
            return render_template('vgg_mammography_finetuned.html', prediction=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Only .png, .jpg and .jpeg are allowed."
            return render_template('vgg_mammography_finetuned.html', prediction=error)

        filename = secure_filename(file.filename)
        images_dir = "./static/images"
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, filename)
        file.save(image_path)

        try:
            img = load_img(image_path, target_size=(224, 224))
            x   = img_to_array(img)
            x   = np.expand_dims(x, axis=0)

            yhat = vgg_mammography_finetuned.model.predict(x, verbose=0)
            prob = float(yhat[0][0])
            TAU  = 0.76
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

        return render_template('vgg_mammography_finetuned.html', prediction=prediction)

    return render_template('vgg_mammography_finetuned.html')


@app.route('/densenet_mammography_initial_unfrozen', methods=['GET', 'POST'])
def densenet_mammography_initial_unfrozen():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
    from tensorflow.keras.models import load_model

    model_path = './model/mammography/initial_unfrozen/mammography_densenet.h5'
    if not hasattr(densenet_mammography_initial_unfrozen, 'model'):
        print("Loading DenseNet121 Mammography Model...")
        densenet_mammography_initial_unfrozen.model = load_model(
            model_path,
            compile=False,
            custom_objects={'preprocess_input': preprocess_input}
        )
        print("Model loaded successfully!")

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            error = "No file part in the request."
            return render_template('densenet_mammography_initial_unfrozen.html', prediction=error)

        file = request.files['imagefile']

        if file.filename == '':
            error = "No file selected."
            return render_template('densenet_mammography_initial_unfrozen.html', prediction=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Only .png, .jpg and .jpeg are allowed."
            return render_template('densenet_mammography_initial_unfrozen.html', prediction=error)

        filename = secure_filename(file.filename)
        images_dir = "./static/images"
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, filename)
        file.save(image_path)

        try:
            img = load_img(image_path, target_size=(224, 224))
            x   = img_to_array(img)
            x   = np.expand_dims(x, axis=0)

            yhat = densenet_mammography_initial_unfrozen.model.predict(x, verbose=0)
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

        return render_template('densenet_mammography_initial_unfrozen.html', prediction=prediction)

    return render_template('densenet_mammography_initial_unfrozen.html')

@app.route('/densenet_mammography_finetuned', methods=['GET', 'POST'])
def densenet_mammography_finetuned():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
    from tensorflow.keras.models import load_model

    model_path = './model/mammography/fine_tune/mammography_densenet.h5'
    if not hasattr(densenet_mammography_finetuned, 'model'):
        print("Loading DenseNet121 Mammography Model...")
        densenet_mammography_finetuned.model = load_model(
            model_path,
            compile=False,
            custom_objects={'preprocess_input': preprocess_input}
        )
        print("Model loaded successfully!")

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            error = "No file part in the request."
            return render_template('densenet_mammography_finetuned.html', prediction=error)

        file = request.files['imagefile']

        if file.filename == '':
            error = "No file selected."
            return render_template('densenet_mammography_finetuned.html', prediction=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Only .png, .jpg and .jpeg are allowed."
            return render_template('densenet_mammography_finetuned.html', prediction=error)

        filename = secure_filename(file.filename)
        images_dir = "./static/images"
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, filename)
        file.save(image_path)

        try:
            img = load_img(image_path, target_size=(224, 224))
            x   = img_to_array(img)
            x   = np.expand_dims(x, axis=0)

            yhat = densenet_mammography_finetuned.model.predict(x, verbose=0)
            prob = float(yhat[0][0])
            TAU  = 0.72
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

        return render_template('densenet_mammography_finetuned.html', prediction=prediction)

    return render_template('densenet_mammography_finetuned.html')


@app.route('/resnet_ultrasound_initial_unfrozen', methods=['GET', 'POST'])
def resnet_ultrasound_initial_unfrozen():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from tensorflow.keras.models import load_model
    
    model_path = './model/ultrasound/initial_unfrozen/ultrasound_resnet.h5'
    if not hasattr(resnet_ultrasound_initial_unfrozen, 'model'):
        print("Loading ResNet50 Ultrasound Model...")
        resnet_ultrasound_initial_unfrozen.model = load_model(
            model_path,
            compile=False,
            custom_objects={'preprocess_input': preprocess_input}
        )
        print("Model loaded successfully!")

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            error = "No file part in the request."
            return render_template('resnet_ultrasound_initial_unfrozen.html', prediction=error)

        file = request.files['imagefile']

        if file.filename == '':
            error = "No file selected."
            return render_template('resnet_ultrasound_initial_unfrozen.html', prediction=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Only .png, .jpg and .jpeg are allowed."
            return render_template('resnet_ultrasound_initial_unfrozen.html', prediction=error)

        filename = secure_filename(file.filename)
        images_dir = "./static/images"
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, filename)
        file.save(image_path)

        try:
            img = load_img(image_path, target_size=(224, 224))
            x   = img_to_array(img)
            x   = np.expand_dims(x, axis=0)

            yhat = resnet_ultrasound_initial_unfrozen.model.predict(x, verbose=0)
            prob = float(yhat[0][0])
            TAU  = 0.43
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

        return render_template('resnet_ultrasound_initial_unfrozen.html', prediction=prediction)

    return render_template('resnet_ultrasound_initial_unfrozen.html')

@app.route('/resnet_ultrasound_finetuned', methods=['GET', 'POST'])
def resnet_ultrasound_finetuned():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from tensorflow.keras.models import load_model
    
    model_path = './model/ultrasound/fine_tune/ultrasound_resnet.h5'
    if not hasattr(resnet_ultrasound_finetuned, 'model'):
        print("Loading ResNet50 Ultrasound Model...")
        resnet_ultrasound_finetuned.model = load_model(
            model_path,
            compile=False,
            custom_objects={'preprocess_input': preprocess_input}
        )
        print("Model loaded successfully!")

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            error = "No file part in the request."
            return render_template('resnet_ultrasound_finetuned.html', prediction=error)

        file = request.files['imagefile']

        if file.filename == '':
            error = "No file selected."
            return render_template('resnet_ultrasound_finetuned.html', prediction=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Only .png, .jpg and .jpeg are allowed."
            return render_template('resnet_ultrasound_finetuned.html', prediction=error)

        filename = secure_filename(file.filename)
        images_dir = "./static/images"
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, filename)
        file.save(image_path)

        try:
            img = load_img(image_path, target_size=(224, 224))
            x   = img_to_array(img)
            x   = np.expand_dims(x, axis=0)

            yhat = resnet_ultrasound_finetuned.model.predict(x, verbose=0)
            prob = float(yhat[0][0])
            TAU  = 0.65
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

        return render_template('resnet_ultrasound_finetuned.html', prediction=prediction)

    return render_template('resnet_ultrasound_finetuned.html')


@app.route('/vgg_ultrasound_initial_unfrozen', methods=['GET', 'POST'])
def vgg_ultrasound_initial_unfrozen():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    from tensorflow.keras.models import load_model

    model_path = './model/ultrasound/initial_unfrozen/ultrasound_vgg.h5'
    if not hasattr(vgg_ultrasound_initial_unfrozen, 'model'):
        print("Loading VGG16 Ultrasound Model...")
        vgg_ultrasound_initial_unfrozen.model = load_model(
            model_path,
            compile=False,
            custom_objects={'preprocess_input': preprocess_input}
        )
        print("Model loaded successfully!")

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            error = "No file part in the request."
            return render_template('vgg_ultrasound_initial_unfrozen.html', prediction=error)

        file = request.files['imagefile']

        if file.filename == '':
            error = "No file selected."
            return render_template('vgg_ultrasound_initial_unfrozen.html', prediction=error)

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

            yhat = vgg_ultrasound_initial_unfrozen.model.predict(x, verbose=0)
            prob = float(yhat[0][0])
            TAU  = 0.61
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

        return render_template('vgg_ultrasound_initial_unfrozen.html', prediction=prediction)

    return render_template('vgg_ultrasound_initial_unfrozen.html')

@app.route('/vgg_ultrasound_finetuned', methods=['GET', 'POST'])
def vgg_ultrasound_finetuned():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    from tensorflow.keras.models import load_model

    model_path = './model/ultrasound/fine_tune/ultrasound_vgg.h5'
    if not hasattr(vgg_ultrasound_finetuned, 'model'):
        print("Loading VGG16 Ultrasound Model...")
        vgg_ultrasound_finetuned.model = load_model(
            model_path,
            compile=False,
            custom_objects={'preprocess_input': preprocess_input}
        )
        print("Model loaded successfully!")

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            error = "No file part in the request."
            return render_template('vgg_ultrasound_finetuned.html', prediction=error)

        file = request.files['imagefile']

        if file.filename == '':
            error = "No file selected."
            return render_template('vgg_ultrasound_finetuned.html', prediction=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Only .png, .jpg and .jpeg are allowed."
            return render_template('vgg_ultrasound_finetuned.html', prediction=error)

        filename = secure_filename(file.filename)
        images_dir = "./static/images"
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, filename)
        file.save(image_path)

        try:
            img = load_img(image_path, target_size=(224, 224))
            x   = img_to_array(img)
            x   = np.expand_dims(x, axis=0)

            yhat = vgg_ultrasound_finetuned.model.predict(x, verbose=0)
            prob = float(yhat[0][0])
            TAU  = 0.7
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

        return render_template('vgg_ultrasound_finetuned.html', prediction=prediction)

    return render_template('vgg_ultrasound_finetuned.html')

@app.route('/densenet_ultrasound_initial_unfrozen', methods=['GET', 'POST'])
def densenet_ultrasound_initial_unfrozen():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
    from tensorflow.keras.models import load_model

    model_path = './model/ultrasound/initial_unfrozen/ultrasound_densenet.h5'
    if not hasattr(densenet_ultrasound_initial_unfrozen, 'model'):
        print("Loading DenseNet121 Ultrasound Model...")
        densenet_ultrasound_initial_unfrozen.model = load_model(
            model_path,
            compile=False,
            custom_objects={'preprocess_input': preprocess_input}
        )
        print("Model loaded successfully!")

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            error = "No file part in the request."
            return render_template('densenet_ultrasound_initial_unfrozen.html', prediction=error)

        file = request.files['imagefile']

        if file.filename == '':
            error = "No file selected."
            return render_template('densenet_ultrasound_initial_unfrozen.html', prediction=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Only .png, .jpg and .jpeg are allowed."
            return render_template('densenet_ultrasound_initial_unfrozen.html', prediction=error)

        filename = secure_filename(file.filename)
        images_dir = "./static/images"
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, filename)
        file.save(image_path)

        try:
            img = load_img(image_path, target_size=(224, 224))
            x   = img_to_array(img)
            x   = np.expand_dims(x, axis=0)

            yhat = densenet_ultrasound_initial_unfrozen.model.predict(x, verbose=0)
            prob = float(yhat[0][0])
            TAU  = 0.70
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

        return render_template('densenet_ultrasound_initial_unfrozen.html', prediction=prediction)

    return render_template('densenet_ultrasound_initial_unfrozen.html')

@app.route('/densenet_ultrasound_finetuned', methods=['GET', 'POST'])
def densenet_ultrasound_finetuned():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
    from tensorflow.keras.models import load_model

    model_path = './model/ultrasound/fine_tune/ultrasound_densenet.h5'
    if not hasattr(densenet_ultrasound_finetuned, 'model'):
        print("Loading DenseNet121 Ultrasound Model...")
        densenet_ultrasound_finetuned.model = load_model(
            model_path,
            compile=False,
            custom_objects={'preprocess_input': preprocess_input}
        )
        print("Model loaded successfully!")

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            error = "No file part in the request."
            return render_template('densenet_ultrasound_finetuned.html', prediction=error)

        file = request.files['imagefile']

        if file.filename == '':
            error = "No file selected."
            return render_template('densenet_ultrasound_finetuned.html', prediction=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Only .png, .jpg and .jpeg are allowed."
            return render_template('densenet_ultrasound_finetuned.html', prediction=error)

        filename = secure_filename(file.filename)
        images_dir = "./static/images"
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, filename)
        file.save(image_path)

        try:
            img = load_img(image_path, target_size=(224, 224))
            x   = img_to_array(img)
            x   = np.expand_dims(x, axis=0)

            yhat = densenet_ultrasound_finetuned.model.predict(x, verbose=0)
            prob = float(yhat[0][0])
            TAU  = 0.71
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

        return render_template('densenet_ultrasound_finetuned.html', prediction=prediction)

    return render_template('densenet_ultrasound_finetuned.html')

@app.route('/resnet_histopathology_initial_unfrozen', methods=['GET', 'POST'])
def resnet_histopathology_initial_unfrozen():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from tensorflow.keras.models import load_model

    model_path = './model/histopathology/initial_unfrozen/histopathology_resnet.h5'
    if not hasattr(resnet_histopathology_initial_unfrozen, 'model'):
        print("Loading ResNet50 Histopathology Model...")
        resnet_histopathology_initial_unfrozen.model = load_model(
            model_path,
            compile=False,
            custom_objects={'preprocess_input': preprocess_input}
        )
        print("Model loaded successfully!")

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            error = "No file part in the request."
            return render_template('resnet_histopathology_initial_frozen.html', prediction=error)

        file = request.files['imagefile']

        if file.filename == '':
            error = "No file selected."
            return render_template('resnet_histopathology_initial_frozen.html', prediction=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Only .png, .jpg and .jpeg are allowed."
            return render_template('resnet_histopathology_initial_frozen.html', prediction=error)

        filename = secure_filename(file.filename)
        images_dir = "./static/images"
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, filename)
        file.save(image_path)

        try:
            img = load_img(image_path, target_size=(224, 224))
            x   = img_to_array(img)
            x   = np.expand_dims(x, axis=0)

            yhat = resnet_histopathology_initial_unfrozen.model.predict(x, verbose=0) 
            prob = float(yhat[0][0])
            TAU  = 0.43
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

        return render_template('resnet_histopathology_initial_unfrozen.html', prediction=prediction)

    return render_template('resnet_histopathology_initial_unfrozen.html')

@app.route('/resnet_histopathology_finetuned', methods=['GET', 'POST'])
def resnet_histopathology_finetuned():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from tensorflow.keras.models import load_model

    model_path = './model/histopathology/fine_tune/histopathology_resnet.h5'
    if not hasattr(resnet_histopathology_finetuned, 'model'):
        print("Loading ResNet50 Histopathology Model...")
        resnet_histopathology_finetuned.model = load_model(
            model_path,
            compile=False,
            custom_objects={'preprocess_input': preprocess_input}
        )
        print("Model loaded successfully!")

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            error = "No file part in the request."
            return render_template('resnet_histopathology_finetuned.html', prediction=error)

        file = request.files['imagefile']

        if file.filename == '':
            error = "No file selected."
            return render_template('resnet_histopathology_finetuned.html', prediction=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Only .png, .jpg and .jpeg are allowed."
            return render_template('resnet_histopathology_finetuned.html', prediction=error)

        filename = secure_filename(file.filename)
        images_dir = "./static/images"
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, filename)
        file.save(image_path)

        try:
            img = load_img(image_path, target_size=(224, 224))
            x   = img_to_array(img)
            x   = np.expand_dims(x, axis=0)

            yhat = resnet_histopathology_finetuned.model.predict(x, verbose=0) 
            prob = float(yhat[0][0])
            TAU  = 0.41
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

        return render_template('resnet_histopathology_finetuned.html', prediction=prediction)

    return render_template('resnet_histopathology_finetuned.html')

@app.route('/vgg_histopathology_initial_unfrozen', methods=['GET', 'POST'])
def vgg_histopathology_initial_unfrozen():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    from tensorflow.keras.models import load_model

    model_path = './model/histopathology/initial_unfrozen/histopathology_vgg.h5'
    if not hasattr(vgg_histopathology_initial_unfrozen, 'model'):
        print("Loading VGG16 Histopathology Model...")
        vgg_histopathology_initial_unfrozen.model = load_model(
            model_path,
            compile=False,
            custom_objects={'preprocess_input': preprocess_input}
        )
        print("Model loaded successfully!")

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            error = "No file part in the request."
            return render_template('vgg_histopathology_initial_unfrozen.html', prediction=error)

        file = request.files['imagefile']

        if file.filename == '':
            error = "No file selected."
            return render_template('vgg_histopathology_initial_unfrozen.html', prediction=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Only .png, .jpg and .jpeg are allowed."
            return render_template('vgg_histopathology_initial_unfrozen.html', prediction=error)

        filename = secure_filename(file.filename)
        images_dir = "./static/images"
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, filename)
        file.save(image_path)

        try:
            img = load_img(image_path, target_size=(224, 224))
            x   = img_to_array(img)
            x   = np.expand_dims(x, axis=0)

            yhat = vgg_histopathology_initial_unfrozen.model.predict(x, verbose=0)
            prob = float(yhat[0][0])
            TAU  = 0.43
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

        return render_template('vgg_histopathology_initial_unfrozen.html', prediction=prediction)

    return render_template('vgg_histopathology_initial_unfrozen.html')

@app.route('/vgg_histopathology_finetuned', methods=['GET', 'POST'])
def vgg_histopathology_finetuned():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    from tensorflow.keras.models import load_model

    model_path = './model/histopathology/fine_tune/histopathology_vgg.h5'
    if not hasattr(vgg_histopathology_finetuned, 'model'):
        print("Loading VGG16 Histopathology Model...")
        vgg_histopathology_finetuned.model = load_model(
            model_path,
            compile=False,
            custom_objects={'preprocess_input': preprocess_input}
        )
        print("Model loaded successfully!")

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            error = "No file part in the request."
            return render_template('vgg_histopathology_finetuned.html', prediction=error)

        file = request.files['imagefile']

        if file.filename == '':
            error = "No file selected."
            return render_template('vgg_histopathology_finetuned.html', prediction=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Only .png, .jpg and .jpeg are allowed."
            return render_template('vgg_histopathology_finetuned.html', prediction=error)

        filename = secure_filename(file.filename)
        images_dir = "./static/images"
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, filename)
        file.save(image_path)

        try:
            img = load_img(image_path, target_size=(224, 224))
            x   = img_to_array(img)
            x   = np.expand_dims(x, axis=0)

            yhat = vgg_histopathology_finetuned.model.predict(x, verbose=0)
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

        return render_template('vgg_histopathology_finetuned.html', prediction=prediction)

    return render_template('vgg_histopathology_finetuned.html')

@app.route('/densenet_histopathology_initial_unfrozen', methods=['GET', 'POST'])
def densenet_histopathology_initial_unfrozen():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from tensorflow.keras.models import load_model

    model_path = './model/histopathology/initial_unfrozen/histopathology_densenet.h5'
    if not hasattr(densenet_histopathology_initial_unfrozen, 'model'):
        print("Loading DenseNet121 Histopathology Model...")
        densenet_histopathology_initial_unfrozen.model = load_model(
            model_path,
            compile=False,
            custom_objects={'preprocess_input': preprocess_input}
        )
        print("Model loaded successfully!")

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            error = "No file part in the request."
            return render_template('densenet_histopathology_initial_unfrozen.html', prediction=error)

        file = request.files['imagefile']

        if file.filename == '':
            error = "No file selected."
            return render_template('densenet_histopathology_initial_unfrozen.html', prediction=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Only .png, .jpg and .jpeg are allowed."
            return render_template('densenet_histopathology_initial_unfrozen.html', prediction=error)

        filename = secure_filename(file.filename)
        images_dir = "./static/images"
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, filename)
        file.save(image_path)

        try:
            img = load_img(image_path, target_size=(224, 224))
            x   = img_to_array(img)
            x   = np.expand_dims(x, axis=0)

            yhat = densenet_histopathology_initial_unfrozen.model.predict(x, verbose=0)
            prob = float(yhat[0][0])
            TAU  = 0.47
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

        return render_template('densenet_histopathology_initial_unfrozen.html', prediction=prediction)

    return render_template('densenet_histopathology_initial_unfrozen.html')

@app.route('/densenet_histopathology_finetuned', methods=['GET', 'POST'])
def densenet_histopathology_finetuned():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    from tensorflow.keras.models import load_model

    model_path = './model/histopathology/fine_tune/histopathology_densenet.h5'
    if not hasattr(densenet_histopathology_finetuned, 'model'):
        print("Loading DenseNet121 Histopathology Model...")
        densenet_histopathology_finetuned.model = load_model(
            model_path,
            compile=False,
            custom_objects={'preprocess_input': preprocess_input}
        )
        print("Model loaded successfully!")

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            error = "No file part in the request."
            return render_template('densenet_histopathology_finetuned.html', prediction=error)

        file = request.files['imagefile']

        if file.filename == '':
            error = "No file selected."
            return render_template('densenet_histopathology_finetuned.html', prediction=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Only .png, .jpg and .jpeg are allowed."
            return render_template('densenet_histopathology_finetuned.html', prediction=error)

        filename = secure_filename(file.filename)
        images_dir = "./static/images"
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, filename)
        file.save(image_path)

        try:
            img = load_img(image_path, target_size=(224, 224))
            x   = img_to_array(img)
            x   = np.expand_dims(x, axis=0)

            yhat = densenet_histopathology_finetuned.model.predict(x, verbose=0)
            prob = float(yhat[0][0])
            TAU  = 0.41
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

        return render_template('densenet_histopathology_finetuned.html', prediction=prediction)

    return render_template('densenet_histopathology_finetuned.html')

@app.route('/mammography_initial_compare', methods=['GET', 'POST'])
def mammography_initial_compare():
    import os
    import numpy as np
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre
    from tensorflow.keras.applications.vgg16    import preprocess_input as vgg_pre
    from tensorflow.keras.applications.densenet import preprocess_input as densenet_pre

    # name → (model_path, its preprocess_fn, its tuned threshold)
    specs = {
        'ResNet50':    ('./model/mammography/initial_unfrozen/mammography_resnet.h5',
                        resnet_pre,    0.39),
        'VGG16':       ('./model/mammography/initial_unfrozen/mammography_vgg.h5',
                        vgg_pre,       0.49),
        'DenseNet121': ('./model/mammography/initial_unfrozen/mammography_densenet.h5',
                        densenet_pre,  0.22),
    }

    # Load all three once, wiring in their own preprocess_input for the Lambda layer
    if not hasattr(mammography_initial_compare, 'models'):
        mammography_initial_compare.models = {}
        for name, (path, pre_fn, _) in specs.items():
            mammography_initial_compare.models[name] = load_model(
                path,
                compile=False,
                custom_objects={'preprocess_input': pre_fn}
            )

    results = {}
    if request.method == 'POST':
        file = request.files.get('imagefile')
        if file and allowed_file(file.filename):
            tmp_path = os.path.join('static/images', secure_filename(file.filename))
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            file.save(tmp_path)

            # Load and resize to (224,224) once
            img = load_img(tmp_path, target_size=(224,224))
            x   = img_to_array(img)[None, ...]    # shape (1,224,224,3)

            # For each model, predict *without* additional preprocessing
            for name, model in mammography_initial_compare.models.items():
                tau  = specs[name][2]
                prob = float(model.predict(x, verbose=0)[0,0])
                label = "Tumor Detected" if prob >= tau else "No Tumor Detected"
                conf  = prob if prob >= tau else 1 - prob
                results[name] = {
                    'label': label,
                    'conf':  f"{conf*100:.1f}%",
                    'tau':   tau
                }

            os.remove(tmp_path)
        else:
            results['error'] = "Please upload a valid PNG/JPG image."

    return render_template('mammography_initial_compare.html', results=results)

# in app.py
@app.route('/mammography_finetuned_compare', methods=['GET', 'POST'])
def mammography_finetuned_compare():
    import os
    import numpy as np
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre
    from tensorflow.keras.applications.vgg16    import preprocess_input as vgg_pre
    from tensorflow.keras.applications.densenet import preprocess_input as densenet_pre

    # name → (model_path, its preprocess_fn, its tuned threshold)
    specs = {
        'ResNet50':    ('./model/mammography/fine_tune/mammography_resnet.h5',
                        resnet_pre,    0.65),
        'VGG16':       ('./model/mammography/fine_tune/mammography_vgg.h5',
                        vgg_pre,       0.76),
        'DenseNet121': ('./model/mammography/fine_tune/mammography_densenet.h5',
                        densenet_pre,  0.72),
    }

    # Load each model once, wiring in its preprocess_input for the Lambda layer
    if not hasattr(mammography_finetuned_compare, 'models'):
        mammography_finetuned_compare.models = {}
        for name, (path, pre_fn, _) in specs.items():
            mammography_finetuned_compare.models[name] = load_model(
                path,
                compile=False,
                custom_objects={'preprocess_input': pre_fn}
            )

    results = {}
    if request.method == 'POST':
        file = request.files.get('imagefile')
        if file and allowed_file(file.filename):
            tmp = os.path.join('static/images', secure_filename(file.filename))
            os.makedirs(os.path.dirname(tmp), exist_ok=True)
            file.save(tmp)

            # load & preprocess once
            img = load_img(tmp, target_size=(224,224))
            x   = img_to_array(img)[None, ...]    # (1,224,224,3)

            for name, model in mammography_finetuned_compare.models.items():
                tau  = specs[name][2]
                # no extra preprocessing: model’s Lambda handles it
                prob = float(model.predict(x, verbose=0)[0,0])
                label = "Tumor Detected" if prob >= tau else "No Tumor Detected"
                conf  = prob if prob >= tau else 1 - prob
                results[name] = {
                    'label': label,
                    'conf':  f"{conf*100:.1f}%",
                    'tau':   tau
                }

            os.remove(tmp)
        else:
            results['error'] = "Please upload a valid PNG/JPG image."

    return render_template('mammography_finetuned_compare.html', results=results)

@app.route('/ultrasound_initial_compare', methods=['GET', 'POST'])
def ultrasound_initial_compare():
    import os
    import numpy as np
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre
    from tensorflow.keras.applications.vgg16    import preprocess_input as vgg_pre
    from tensorflow.keras.applications.densenet import preprocess_input as densenet_pre

    # name → (model_path, its preprocess_fn, its tuned threshold)
    specs = {
        'ResNet50':    ('./model/ultrasound/initial_unfrozen/ultrasound_resnet.h5',
                        resnet_pre,    0.43),
        'VGG16':       ('./model/ultrasound/initial_unfrozen/ultrasound_vgg.h5',
                        vgg_pre,       0.61),
        'DenseNet121': ('./model/ultrasound/initial_unfrozen/ultrasound_densenet.h5',
                        densenet_pre,  0.70),
    }

    # Load each model once, wiring in its preprocess_input for the Lambda layer
    if not hasattr(ultrasound_initial_compare, 'models'):
        ultrasound_initial_compare.models = {}
        for name, (path, pre_fn, _) in specs.items():
            ultrasound_initial_compare.models[name] = load_model(
                path,
                compile=False,
                custom_objects={'preprocess_input': pre_fn}
            )

    results = {}
    if request.method == 'POST':
        file = request.files.get('imagefile')
        if file and allowed_file(file.filename):
            tmp = os.path.join('static/images', secure_filename(file.filename))
            os.makedirs(os.path.dirname(tmp), exist_ok=True)
            file.save(tmp)

            # load & preprocess once
            img = load_img(tmp, target_size=(224,224))
            x   = img_to_array(img)[None, ...]    # (1,224,224,3)

            for name, model in ultrasound_initial_compare.models.items():
                tau  = specs[name][2]
                # no extra preprocessing: model’s Lambda handles it
                prob = float(model.predict(x, verbose=0)[0,0])
                label = "Tumor Detected" if prob >= tau else "No Tumor Detected"
                conf  = prob if prob >= tau else 1 - prob
                results[name] = {
                    'label': label,
                    'conf':  f"{conf*100:.1f}%",
                    'tau':   tau
                }

            os.remove(tmp)
        else:
            results['error'] = "Please upload a valid PNG/JPG image."

    return render_template('ultrasound_initial_compare.html', results=results)


@app.route('/ultrasound_finetuned_compare', methods=['GET', 'POST'])
def ultrasound_finetuned_compare():
    import os
    import numpy as np
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre
    from tensorflow.keras.applications.vgg16    import preprocess_input as vgg_pre
    from tensorflow.keras.applications.densenet import preprocess_input as densenet_pre

    # name → (model_path, its preprocess_fn, its tuned threshold)
    specs = {
        'ResNet50':    ('./model/ultrasound/fine_tune/ultrasound_resnet.h5',
                        resnet_pre,    0.65),
        'VGG16':       ('./model/ultrasound/fine_tune/ultrasound_vgg.h5',
                        vgg_pre,       0.70),
        'DenseNet121': ('./model/ultrasound/fine_tune/ultrasound_densenet.h5',
                        densenet_pre,  0.71),
    }

    # Load each model once, wiring in its preprocess_input for the Lambda layer
    if not hasattr(ultrasound_finetuned_compare, 'models'):
        ultrasound_finetuned_compare.models = {}
        for name, (path, pre_fn, _) in specs.items():
            ultrasound_finetuned_compare.models[name] = load_model(
                path,
                compile=False,
                custom_objects={'preprocess_input': pre_fn}
            )

    results = {}
    if request.method == 'POST':
        file = request.files.get('imagefile')
        if file and allowed_file(file.filename):
            tmp = os.path.join('static/images', secure_filename(file.filename))
            os.makedirs(os.path.dirname(tmp), exist_ok=True)
            file.save(tmp)

            # load & preprocess once
            img = load_img(tmp, target_size=(224,224))
            x   = img_to_array(img)[None, ...]    # (1,224,224,3)

            for name, model in ultrasound_finetuned_compare.models.items():
                tau  = specs[name][2]
                # no extra preprocessing: model’s Lambda handles it
                prob = float(model.predict(x, verbose=0)[0,0])
                label = "Tumor Detected" if prob >= tau else "No Tumor Detected"
                conf  = prob if prob >= tau else 1 - prob
                results[name] = {
                    'label': label,
                    'conf':  f"{conf*100:.1f}%",
                    'tau':   tau
                }

            os.remove(tmp)
        else:
            results['error'] = "Please upload a valid PNG/JPG image."

    return render_template('ultrasound_finetuned_compare.html', results=results)

@app.route('/histopathology_initial_compare', methods=['GET', 'POST'])
def histopathology_initial_compare():
    import os
    import numpy as np
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre
    from tensorflow.keras.applications.vgg16    import preprocess_input as vgg_pre
    from tensorflow.keras.applications.densenet import preprocess_input as densenet_pre

    # name → (model_path, its preprocess_fn, its tuned threshold)
    specs = {
        'ResNet50':    ('./model/histopathology/initial_unfrozen/histopathology_resnet.h5',    resnet_pre, 0.43),
        'VGG16':       ('./model/histopathology/initial_unfrozen/histopathology_vgg.h5',       vgg_pre,    0.43),
        'DenseNet121': ('./model/histopathology/initial_unfrozen/histopathology_densenet.h5', densenet_pre, 0.49),
    }

    if not hasattr(histopathology_initial_compare, 'models'):
        histopathology_initial_compare.models = {}
        for name, (path, pre_fn, _) in specs.items():
            histopathology_initial_compare.models[name] = load_model(
                path,
                compile=False,
                custom_objects={'preprocess_input': pre_fn}
            )

    results = {}
    if request.method == 'POST':
        file = request.files.get('imagefile')
        if file and allowed_file(file.filename):
            tmp = os.path.join('static/images', secure_filename(file.filename))
            os.makedirs(os.path.dirname(tmp), exist_ok=True)
            file.save(tmp)

            img = load_img(tmp, target_size=(224,224))
            x   = img_to_array(img)[None, ...]   # (1,224,224,3)

            for name, model in histopathology_initial_compare.models.items():
                tau  = specs[name][2]
                prob = float(model.predict(x, verbose=0)[0,0])
                label = "Tumor Detected" if prob >= tau else "No Tumor Detected"
                conf  = prob if prob >= tau else 1 - prob
                results[name] = {'label': label, 'conf': f"{conf*100:.1f}%", 'tau': tau}

            os.remove(tmp)
        else:
            results['error'] = "Please upload a valid PNG/JPG image."

    return render_template('histopathology_initial_compare.html', results=results)


@app.route('/histopathology_finetuned_compare', methods=['GET', 'POST'])
def histopathology_finetuned_compare():
    import os
    import numpy as np
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre
    from tensorflow.keras.applications.vgg16    import preprocess_input as vgg_pre
    from tensorflow.keras.applications.densenet import preprocess_input as densenet_pre

    specs = {
        'ResNet50':    ('./model/histopathology/fine_tune/histopathology_resnet.h5',    resnet_pre, 0.41),
        'VGG16':       ('./model/histopathology/fine_tune/histopathology_vgg.h5',       vgg_pre,    0.46),
        'DenseNet121': ('./model/histopathology/fine_tune/histopathology_densenet.h5', densenet_pre, 0.50),
    }

    if not hasattr(histopathology_finetuned_compare, 'models'):
        histopathology_finetuned_compare.models = {}
        for name, (path, pre_fn, _) in specs.items():
            histopathology_finetuned_compare.models[name] = load_model(
                path,
                compile=False,
                custom_objects={'preprocess_input': pre_fn}
            )

    results = {}
    if request.method == 'POST':
        file = request.files.get('imagefile')
        if file and allowed_file(file.filename):
            tmp = os.path.join('static/images', secure_filename(file.filename))
            os.makedirs(os.path.dirname(tmp), exist_ok=True)
            file.save(tmp)

            img = load_img(tmp, target_size=(224,224))
            x   = img_to_array(img)[None, ...]

            for name, model in histopathology_finetuned_compare.models.items():
                tau  = specs[name][2]
                prob = float(model.predict(x, verbose=0)[0,0])
                label = "Tumor Detected" if prob >= tau else "No Tumor Detected"
                conf  = prob if prob >= tau else 1 - prob
                results[name] = {'label': label, 'conf': f"{conf*100:.1f}%", 'tau': tau}

            os.remove(tmp)
        else:
            results['error'] = "Please upload a valid PNG/JPG image."

    return render_template('histopathology_finetuned_compare.html', results=results)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=True)