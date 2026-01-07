"""Centralized configuration for model registry, thresholds, and constants."""

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMAGE_SIZE = (224, 224)
IMAGES_DIR = './static/images'

ARCHITECTURES = {
    'resnet': {'name': 'ResNet50', 'image': 'images/resnet_architecture.png'},
    'vgg': {'name': 'VGG16', 'image': 'images/vgg_architecture.png'},
    'densenet': {'name': 'DenseNet121', 'image': 'images/densenet_architecture.jpeg'}
}

MODALITIES = {
    'mammography': {'name': 'Mammography'},
    'ultrasound': {'name': 'Ultrasound'},
    'histopathology': {'name': 'Histopathology'}
}

TRAINING_TYPES = {
    'initial': {'name': 'Initial Unfrozen', 'folder': 'initial_unfrozen'},
    'finetuned': {'name': 'Fine-tuned', 'folder': 'fine_tune'}
}

MODEL_CONFIG = {
    # Mammography - ResNet
    ('mammography', 'resnet', 'initial'): {
        'path': './model/mammography/initial_unfrozen/mammography_resnet.h5',
        'threshold': 0.39,
        'preprocess': 'resnet'
    },
    ('mammography', 'resnet', 'finetuned'): {
        'path': './model/mammography/fine_tune/mammography_resnet.h5',
        'threshold': 0.65,
        'preprocess': 'resnet'
    },
    # Mammography - VGG
    ('mammography', 'vgg', 'initial'): {
        'path': './model/mammography/initial_unfrozen/mammography_vgg.h5',
        'threshold': 0.49,
        'preprocess': 'vgg'
    },
    ('mammography', 'vgg', 'finetuned'): {
        'path': './model/mammography/fine_tune/mammography_vgg.h5',
        'threshold': 0.76,
        'preprocess': 'vgg'
    },
    # Mammography - DenseNet
    ('mammography', 'densenet', 'initial'): {
        'path': './model/mammography/initial_unfrozen/mammography_densenet.h5',
        'threshold': 0.22,
        'preprocess': 'densenet'
    },
    ('mammography', 'densenet', 'finetuned'): {
        'path': './model/mammography/fine_tune/mammography_densenet.h5',
        'threshold': 0.72,
        'preprocess': 'densenet'
    },
    # Ultrasound - ResNet
    ('ultrasound', 'resnet', 'initial'): {
        'path': './model/ultrasound/initial_unfrozen/ultrasound_resnet.h5',
        'threshold': 0.43,
        'preprocess': 'resnet'
    },
    ('ultrasound', 'resnet', 'finetuned'): {
        'path': './model/ultrasound/fine_tune/ultrasound_resnet.h5',
        'threshold': 0.65,
        'preprocess': 'resnet'
    },
    # Ultrasound - VGG
    ('ultrasound', 'vgg', 'initial'): {
        'path': './model/ultrasound/initial_unfrozen/ultrasound_vgg.h5',
        'threshold': 0.61,
        'preprocess': 'vgg'
    },
    ('ultrasound', 'vgg', 'finetuned'): {
        'path': './model/ultrasound/fine_tune/ultrasound_vgg.h5',
        'threshold': 0.70,
        'preprocess': 'vgg'
    },
    # Ultrasound - DenseNet
    ('ultrasound', 'densenet', 'initial'): {
        'path': './model/ultrasound/initial_unfrozen/ultrasound_densenet.h5',
        'threshold': 0.70,
        'preprocess': 'densenet'
    },
    ('ultrasound', 'densenet', 'finetuned'): {
        'path': './model/ultrasound/fine_tune/ultrasound_densenet.h5',
        'threshold': 0.71,
        'preprocess': 'densenet'
    },
    # Histopathology - ResNet
    ('histopathology', 'resnet', 'initial'): {
        'path': './model/histopathology/initial_unfrozen/histopathology_resnet.h5',
        'threshold': 0.43,
        'preprocess': 'resnet'
    },
    ('histopathology', 'resnet', 'finetuned'): {
        'path': './model/histopathology/fine_tune/histopathology_resnet.h5',
        'threshold': 0.41,
        'preprocess': 'resnet'
    },
    # Histopathology - VGG
    ('histopathology', 'vgg', 'initial'): {
        'path': './model/histopathology/initial_unfrozen/histopathology_vgg.h5',
        'threshold': 0.43,
        'preprocess': 'vgg'
    },
    ('histopathology', 'vgg', 'finetuned'): {
        'path': './model/histopathology/fine_tune/histopathology_vgg.h5',
        'threshold': 0.46,
        'preprocess': 'vgg'
    },
    # Histopathology - DenseNet
    ('histopathology', 'densenet', 'initial'): {
        'path': './model/histopathology/initial_unfrozen/histopathology_densenet.h5',
        'threshold': 0.47,
        'preprocess': 'densenet'
    },
    ('histopathology', 'densenet', 'finetuned'): {
        'path': './model/histopathology/fine_tune/histopathology_densenet.h5',
        'threshold': 0.41,
        'preprocess': 'densenet'
    },
}


def get_model_config(modality, architecture, training_type):
    """Return config for a specific model variant."""
    key = (modality, architecture, training_type)
    if key not in MODEL_CONFIG:
        raise ValueError(f"Unknown model config: {key}")
    return MODEL_CONFIG[key]


def get_image_paths(modality, architecture, training_type):
    """Return paths for model visualization images."""
    folder = TRAINING_TYPES[training_type]['folder']
    prefix = f"{modality}_{architecture}"
    base = f"images/{modality}/{folder}"
    return {
        'confusion_matrix': f"{base}/{prefix}_confusion_matrix.png",
        'learning': f"{base}/{prefix}_learning.png",
        'roc_curve': f"{base}/{prefix}_roc_curve.png"
    }


def get_display_info(modality, architecture, training_type):
    """Return display names and title for a model variant."""
    return {
        'modality_name': MODALITIES[modality]['name'],
        'arch_name': ARCHITECTURES[architecture]['name'],
        'training_name': TRAINING_TYPES[training_type]['name'],
        'arch_image': ARCHITECTURES[architecture]['image'],
        'title': f"{ARCHITECTURES[architecture]['name']} {TRAINING_TYPES[training_type]['name']} - {MODALITIES[modality]['name']}"
    }
