"""Prediction routes with generic handlers."""
from flask import Blueprint, render_template, request

from config import MODALITIES, get_display_info, get_image_paths
from src.services.predictor import predict_single
from src.utils.image_processing import allowed_file, save_temp_image, cleanup_image
from src.data.metrics import get_model_metrics

prediction_bp = Blueprint('prediction', __name__)


@prediction_bp.route('/predict/<modality>/<architecture>/<training_type>', methods=['GET', 'POST'])
def predict(modality, architecture, training_type):
    """Generic prediction route for all model variants."""
    display = get_display_info(modality, architecture, training_type)
    images = get_image_paths(modality, architecture, training_type)
    metrics_data = get_model_metrics(modality, architecture, training_type)
    
    prediction = None
    
    if request.method == 'POST':
        if 'imagefile' not in request.files:
            prediction = "No file part in the request."
        else:
            file = request.files['imagefile']
            if file.filename == '':
                prediction = "No file selected."
            elif not allowed_file(file.filename):
                prediction = "Invalid file type. Only .png, .jpg and .jpeg are allowed."
            else:
                image_path = save_temp_image(file)
                try:
                    result = predict_single(image_path, modality, architecture, training_type)
                    prediction = result['display']
                finally:
                    cleanup_image(image_path)
    
    return render_template('predict.html',
        modality=modality,
        architecture=architecture,
        training_type=training_type,
        title=display['title'],
        modality_name=display['modality_name'],
        arch_name=display['arch_name'],
        training_name=display['training_name'],
        arch_image=display['arch_image'],
        images=images,
        metrics_text=metrics_data.get('metrics', ''),
        report_text=metrics_data.get('report', ''),
        about_text=metrics_data.get('about', ''),
        prediction=prediction
    )


@prediction_bp.route('/mammography')
def mammography():
    """Mammography modality page."""
    return render_template('modality.html',
        modality='mammography',
        modality_name=MODALITIES['mammography']['name']
    )


@prediction_bp.route('/ultrasound')
def ultrasound():
    """Ultrasound modality page."""
    return render_template('modality.html',
        modality='ultrasound',
        modality_name=MODALITIES['ultrasound']['name']
    )


@prediction_bp.route('/histopathology')
def histopathology():
    """Histopathology modality page."""
    return render_template('modality.html',
        modality='histopathology',
        modality_name=MODALITIES['histopathology']['name']
    )
