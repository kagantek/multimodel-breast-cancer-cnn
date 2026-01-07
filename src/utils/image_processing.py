"""Image processing utilities."""
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

from config import ALLOWED_EXTENSIONS, IMAGE_SIZE, IMAGES_DIR


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_temp_image(file):
    """Save uploaded file to temp location and return path."""
    filename = secure_filename(file.filename)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    path = os.path.join(IMAGES_DIR, filename)
    file.save(path)
    return path


def load_and_preprocess(image_path):
    """Load image and preprocess for model input."""
    img = load_img(image_path, target_size=IMAGE_SIZE)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x


def cleanup_image(image_path):
    """Remove temporary image file."""
    if os.path.exists(image_path):
        os.remove(image_path)
