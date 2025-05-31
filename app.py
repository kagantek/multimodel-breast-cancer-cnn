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

@app.route('/histopathology')
def histopathology():
    return render_template('histopathology.html')

@app.route('/about')
def about():
    return render_template('about.html')

"""
@app.route('/resnet_mammography')
def resnet_mammography():
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
    return render_template('densenet_histopathology.html') """

if __name__ == '__main__':
    app.run(debug=True)