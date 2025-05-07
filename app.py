from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
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


app.run(debug=True)