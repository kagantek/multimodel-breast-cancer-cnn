"""Model loading service with caching."""
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

from config import MODEL_CONFIG

PREPROCESS_FUNCTIONS = {
    'resnet': resnet_preprocess,
    'vgg': vgg_preprocess,
    'densenet': densenet_preprocess
}

_model_cache = {}


def get_preprocess_function(architecture):
    """Return preprocessing function for architecture."""
    return PREPROCESS_FUNCTIONS.get(architecture)


def load_cached_model(modality, architecture, training_type):
    """Load model from cache or disk."""
    key = (modality, architecture, training_type)
    
    if key not in _model_cache:
        config = MODEL_CONFIG.get(key)
        if not config:
            raise ValueError(f"Unknown model: {key}")
        
        preprocess_fn = get_preprocess_function(config['preprocess'])
        _model_cache[key] = load_model(
            config['path'],
            compile=False,
            custom_objects={'preprocess_input': preprocess_fn}
        )
    
    return _model_cache[key]


def get_threshold(modality, architecture, training_type):
    """Return threshold for a model variant."""
    config = MODEL_CONFIG.get((modality, architecture, training_type))
    if not config:
        raise ValueError(f"Unknown model config")
    return config['threshold']
