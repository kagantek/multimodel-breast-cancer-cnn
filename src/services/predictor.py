"""Prediction service."""
from src.services.model_loader import load_cached_model, get_threshold
from src.utils.image_processing import load_and_preprocess


def predict_single(image_path, modality, architecture, training_type):
    """Run prediction on a single image."""
    model = load_cached_model(modality, architecture, training_type)
    threshold = get_threshold(modality, architecture, training_type)
    
    x = load_and_preprocess(image_path)
    prob = float(model.predict(x, verbose=0)[0][0])
    
    if prob >= threshold:
        label = "Tumor Detected"
        confidence = prob
    else:
        label = "No Tumor Detected"
        confidence = 1 - prob
    
    return {
        'label': label,
        'confidence': confidence,
        'probability': prob,
        'threshold': threshold,
        'display': f"{label} ({confidence*100:.2f}%) Threshold: {threshold}"
    }


def predict_comparison(image_path, modality, training_type):
    """Run prediction on all architectures for comparison."""
    architectures = ['resnet', 'vgg', 'densenet']
    results = {}
    
    for arch in architectures:
        model = load_cached_model(modality, arch, training_type)
        threshold = get_threshold(modality, arch, training_type)
        
        x = load_and_preprocess(image_path)
        prob = float(model.predict(x, verbose=0)[0][0])
        
        if prob >= threshold:
            label = "Tumor Detected"
            conf = prob
        else:
            label = "No Tumor Detected"
            conf = 1 - prob
        
        arch_names = {'resnet': 'ResNet50', 'vgg': 'VGG16', 'densenet': 'DenseNet121'}
        results[arch_names[arch]] = {
            'label': label,
            'conf': f"{conf*100:.1f}%",
            'tau': threshold
        }
    
    return results
