from flask import Flask, render_template, request
import numpy as np

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
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from tensorflow.keras.models import load_model
    
    print("Loading ResNet50 Mammography Model...")
    modelResnet = load_model('./model/mammography/mammography_resnet.h5', compile=False, custom_objects={'preprocess_input': preprocess_input})
    print("Model loaded successfully!")

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

        yhat = modelResnet.predict(image, verbose=0)
        prob = float(yhat[0][0])
        TAU = 0.39
        label = "Tumor Detected" if prob >= TAU else "No Tumor Detected"
        conf = prob if prob >= TAU else 1 - prob
        prediction = f"{label} ({conf*100:.2f}%) (Threshold: {TAU})"

        return render_template('resnet_mammography.html', prediction=prediction)

    return render_template('resnet_mammography.html')


@app.route('/vgg_mammography', methods=['GET', 'POST'])
def vgg_mammography():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    from tensorflow.keras.models import load_model

    print("Loading VGG16 Mammography Model...")
    modelVgg = load_model('./model/mammography/mammography_vgg.h5', compile=False, custom_objects={'preprocess_input': preprocess_input})
    print("Model loaded successfully!")
    
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

        yhat = modelVgg.predict(image, verbose=0)
        prob = float(yhat[0][0])
        TAU = 0.49
        label = "Tumor Detected" if prob >= TAU else "No Tumor Detected"
        conf = prob if prob >= TAU else 1 - prob
        prediction = f"{label} ({conf*100:.2f}%) (Threshold: {TAU})"

        return render_template('vgg_mammography.html', prediction=prediction)

    return render_template('vgg_mammography.html')


@app.route('/densenet_mammography', methods=['GET', 'POST'])
def densenet_mammography():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
    from tensorflow.keras.models import load_model
    
    print("Loading DenseNet121 Mammography Model...")
    modelDensenet = load_model('./model/mammography/mammography_densenet.h5', compile=False, custom_objects={'preprocess_input': preprocess_input})
    print("Model loaded successfully!")
    
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

        yhat = modelDensenet.predict(image, verbose=0)
        probability = float(yhat[0][0])

        prob = float(modelDensenet.predict(image, verbose=0)[0][0])
        TAU  = 0.22
        label = "Tumor Detected" if prob >= TAU else "No Tumor Detected"
        conf  = prob if prob >= TAU else 1 - prob
        prediction = f"{label} ({conf*100:.2f}%) (Threshold: {TAU})"
        
        return render_template('densenet_mammography.html', prediction=prediction)

    return render_template('densenet_mammography.html')


@app.route('/resnet_ultrasound', methods=['GET', 'POST'])
def resnet_ultrasound():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from tensorflow.keras.models import load_model
    
    print("Loading ResNet50 Ultrasound Model...")
    modelResnet = load_model('./model/ultrasound/ultrasound_resnet.h5', compile=False, custom_objects={'preprocess_input': preprocess_input})
    print("Model loaded successfully!")

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

        yhat = modelResnet.predict(image, verbose=0)
        prob = float(yhat[0][0])
        TAU = 0.49
        label = "Tumor Detected" if prob >= TAU else "No Tumor Detected"
        conf = prob if prob >= TAU else 1 - prob
        prediction = f"{label} ({conf*100:.2f}%) (Threshold: {TAU})"

        return render_template('resnet_ultrasound.html', prediction=prediction)

    return render_template('resnet_ultrasound.html')

@app.route('/vgg_ultrasound', methods=['GET', 'POST'])
def vgg_ultrasound():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    from tensorflow.keras.models import load_model

    print("Loading VGG16 Ultrasound Model...")
    modelVgg = load_model('./model/ultrasound/ultrasound_vgg.h5', compile=False, custom_objects={'preprocess_input': preprocess_input})
    print("Model loaded successfully!")
    
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

        yhat = modelVgg.predict(image, verbose=0)
        prob = float(yhat[0][0])
        TAU = 0.49
        label = "Tumor Detected" if prob >= TAU else "No Tumor Detected"
        conf = prob if prob >= TAU else 1 - prob
        prediction = f"{label} ({conf*100:.2f}%) (Threshold: {TAU})"

        return render_template('vgg_ultrasound.html', prediction=prediction)

    return render_template('vgg_ultrasound.html')

@app.route('/densenet_ultrasound', methods=['GET', 'POST'])
def densenet_ultrasound():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
    from tensorflow.keras.models import load_model

    print("Loading DenseNet121 Ultrasound Model...")
    modelDensenet = load_model('./model/ultrasound/ultrasound_densenet.h5', compile=False, custom_objects={'preprocess_input': preprocess_input})
    print("Model loaded successfully!")
    
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

        yhat = modelDensenet.predict(image, verbose=0)
        probability = float(yhat[0][0])

        prob = float(modelDensenet.predict(image, verbose=0)[0][0])
        TAU  = 0.22
        label = "Tumor Detected" if prob >= TAU else "No Tumor Detected"
        conf  = prob if prob >= TAU else 1 - prob
        prediction = f"{label} ({conf*100:.2f}%) (Threshold: {TAU})"

        return render_template('densenet_ultrasound.html', prediction=prediction)

    return render_template('densenet_ultrasound.html')

@app.route('/resnet_histopathology', methods=['GET', 'POST'])
def resnet_histopathology():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from tensorflow.keras.models import load_model

    print("Loading ResNet50 Histopathology Model...")
    modelResnet = load_model('./model/histopathology/histopathology_resnet.h5', compile=False, custom_objects={'preprocess_input': preprocess_input})
    print("Model loaded successfully!")
    
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

        yhat = modelResnet.predict(image, verbose=0)
        probability = float(yhat[0][0])

        prob = float(modelResnet.predict(image, verbose=0)[0][0])
        TAU  = 0.46
        label = "Tumor Detected" if prob >= TAU else "No Tumor Detected"
        conf  = prob if prob >= TAU else 1 - prob
        prediction = f"{label} ({conf*100:.2f}%) (Threshold: {TAU})"

        return render_template('resnet_histopathology.html', prediction=prediction)

    return render_template('resnet_histopathology.html')

@app.route('/vgg_histopathology', methods=['GET', 'POST'])
def vgg_histopathology():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    from tensorflow.keras.models import load_model

    print("Loading VGG16 Histopathology Model...")
    modelVgg = load_model('./model/histopathology/histopathology_vgg.h5', compile=False, custom_objects={'preprocess_input': preprocess_input})
    print("Model loaded successfully!")
    
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

        yhat = modelVgg.predict(image, verbose=0)
        probability = float(yhat[0][0])

        prob = float(modelVgg.predict(image, verbose=0)[0][0])
        TAU  = 0.50
        label = "Tumor Detected" if prob >= TAU else "No Tumor Detected"
        conf  = prob if prob >= TAU else 1 - prob
        prediction = f"{label} ({conf*100:.2f}%) (Threshold: {TAU})"

        return render_template('vgg_histopathology.html', prediction=prediction)

    return render_template('vgg_histopathology.html')

@app.route('/densenet_histopathology', methods=['GET', 'POST'])
def densenet_histopathology():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
    from tensorflow.keras.models import load_model

    print("Loading DenseNet121 Histopathology Model...")
    modelDenseNet = load_model('./model/histopathology/histopathology_densenet121.h5', compile=False, custom_objects={'preprocess_input': preprocess_input})
    print("Model loaded successfully!")
    
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

        yhat = modelDenseNet.predict(image, verbose=0)
        probability = float(yhat[0][0])

        prob = float(modelDenseNet.predict(image, verbose=0)[0][0])
        TAU  = 0.46
        label = "Tumor Detected" if prob >= TAU else "No Tumor Detected"
        conf  = prob if prob >= TAU else 1 - prob
        prediction = f"{label} ({conf*100:.2f}%) (Threshold: {TAU})"

        return render_template('densenet_histopathology.html', prediction=prediction)

    return render_template('densenet_histopathology.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=True)