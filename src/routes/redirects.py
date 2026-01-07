"""Backward compatibility redirects from old URLs to new URLs."""
from flask import Blueprint, redirect, url_for

redirects_bp = Blueprint('redirects', __name__)

OLD_ROUTE_MAP = {
    'resnet_mammography_initial_unfrozen': ('mammography', 'resnet', 'initial'),
    'resnet_mammography_finetuned': ('mammography', 'resnet', 'finetuned'),
    'vgg_mammography_initial_unfrozen': ('mammography', 'vgg', 'initial'),
    'vgg_mammography_finetuned': ('mammography', 'vgg', 'finetuned'),
    'densenet_mammography_initial_unfrozen': ('mammography', 'densenet', 'initial'),
    'densenet_mammography_finetuned': ('mammography', 'densenet', 'finetuned'),
    'resnet_ultrasound_initial_unfrozen': ('ultrasound', 'resnet', 'initial'),
    'resnet_ultrasound_finetuned': ('ultrasound', 'resnet', 'finetuned'),
    'vgg_ultrasound_initial_unfrozen': ('ultrasound', 'vgg', 'initial'),
    'vgg_ultrasound_finetuned': ('ultrasound', 'vgg', 'finetuned'),
    'densenet_ultrasound_initial_unfrozen': ('ultrasound', 'densenet', 'initial'),
    'densenet_ultrasound_finetuned': ('ultrasound', 'densenet', 'finetuned'),
    'resnet_histopathology_initial_unfrozen': ('histopathology', 'resnet', 'initial'),
    'resnet_histopathology_finetuned': ('histopathology', 'resnet', 'finetuned'),
    'vgg_histopathology_initial_unfrozen': ('histopathology', 'vgg', 'initial'),
    'vgg_histopathology_finetuned': ('histopathology', 'vgg', 'finetuned'),
    'densenet_histopathology_initial_unfrozen': ('histopathology', 'densenet', 'initial'),
    'densenet_histopathology_finetuned': ('histopathology', 'densenet', 'finetuned'),
}

COMPARE_ROUTE_MAP = {
    'mammography_initial_compare': ('mammography', 'initial'),
    'mammography_finetuned_compare': ('mammography', 'finetuned'),
    'ultrasound_initial_compare': ('ultrasound', 'initial'),
    'ultrasound_finetuned_compare': ('ultrasound', 'finetuned'),
    'histopathology_initial_compare': ('histopathology', 'initial'),
    'histopathology_finetuned_compare': ('histopathology', 'finetuned'),
}


def create_prediction_redirect(old_route):
    """Create redirect function for old prediction routes."""
    modality, arch, training = OLD_ROUTE_MAP[old_route]
    def redirect_func():
        return redirect(url_for('prediction.predict', 
            modality=modality, architecture=arch, training_type=training), code=301)
    redirect_func.__name__ = old_route
    return redirect_func


def create_compare_redirect(old_route):
    """Create redirect function for old compare routes."""
    modality, training = COMPARE_ROUTE_MAP[old_route]
    def redirect_func():
        return redirect(url_for('comparison.compare', 
            modality=modality, training_type=training), code=301)
    redirect_func.__name__ = old_route
    return redirect_func


for route_name in OLD_ROUTE_MAP:
    redirects_bp.add_url_rule(
        f'/{route_name}',
        route_name,
        create_prediction_redirect(route_name),
        methods=['GET', 'POST']
    )

for route_name in COMPARE_ROUTE_MAP:
    redirects_bp.add_url_rule(
        f'/{route_name}',
        route_name,
        create_compare_redirect(route_name),
        methods=['GET', 'POST']
    )
