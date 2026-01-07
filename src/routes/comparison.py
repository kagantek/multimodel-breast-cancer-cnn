"""Comparison routes for side-by-side model evaluation."""
from flask import Blueprint, render_template, request

from config import MODALITIES, TRAINING_TYPES
from src.services.predictor import predict_comparison
from src.utils.image_processing import allowed_file, save_temp_image, cleanup_image

comparison_bp = Blueprint('comparison', __name__)


@comparison_bp.route('/compare/<modality>/<training_type>', methods=['GET', 'POST'])
def compare(modality, training_type):
    """Generic comparison route for all modality/training combinations."""
    modality_name = MODALITIES[modality]['name']
    training_name = TRAINING_TYPES[training_type]['name']
    
    results = None
    error = None
    
    if request.method == 'POST':
        file = request.files.get('imagefile')
        if file and allowed_file(file.filename):
            image_path = save_temp_image(file)
            try:
                results = predict_comparison(image_path, modality, training_type)
            finally:
                cleanup_image(image_path)
        else:
            error = "Please upload a valid PNG/JPG image."
    
    return render_template('compare.html',
        modality=modality,
        training_type=training_type,
        modality_name=modality_name,
        training_name=training_name,
        results=results,
        error=error
    )
