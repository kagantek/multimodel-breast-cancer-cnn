"""Main routes for static pages."""
from flask import Blueprint, render_template

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
@main_bp.route('/index')
def index():
    return render_template('index.html')


@main_bp.route('/about')
def about():
    return render_template('about.html')


@main_bp.route('/notebook')
def notebook():
    return render_template('notebook.html')


@main_bp.route('/comparison')
def comparison():
    return render_template('comparison.html')


@main_bp.route('/mammography_notebook')
def mammography_notebook():
    return render_template('mammography_notebook.html')


@main_bp.route('/ultrasound_notebook')
def ultrasound_notebook():
    return render_template('ultrasound_notebook.html')


@main_bp.route('/histopathology_notebook')
def histopathology_notebook():
    return render_template('histopathology_notebook.html')
