from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/mammography')
def mammography():
    return render_template('mammography.html')

@app.route('/ultrasound')
def ultrasound():
    return render_template('ultrasound.html')

@app.route('/mri')
def mri():
    return render_template('mri.html')

@app.route('/resnet_mammography')
def resnet_mammography():
    return render_template('resnet_mammography.html')

@app.route('/vgg_mammography')
def vgg_mammography():
    return render_template('vgg_mammography.html')

@app.route('/alexnet_mammography')
def alexnet_mammography():
    return render_template('alexnet_mammography.html')

app.run(debug=True)