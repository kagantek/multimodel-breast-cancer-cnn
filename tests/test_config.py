import pytest
from config import (
    MODEL_CONFIG, MODALITIES, ARCHITECTURES, TRAINING_TYPES,
    get_model_config, get_image_paths, get_display_info,
    ALLOWED_EXTENSIONS, IMAGE_SIZE
)


class TestConstants:
    def test_allowed_extensions(self):
        assert 'png' in ALLOWED_EXTENSIONS
        assert 'jpg' in ALLOWED_EXTENSIONS
        assert 'jpeg' in ALLOWED_EXTENSIONS
    
    def test_image_size(self):
        assert IMAGE_SIZE == (224, 224)


class TestModalities:
    def test_all_modalities_present(self):
        expected = {'mammography', 'ultrasound', 'histopathology'}
        assert set(MODALITIES.keys()) == expected
    
    def test_modality_has_name(self):
        for mod in MODALITIES.values():
            assert 'name' in mod
            assert isinstance(mod['name'], str)


class TestArchitectures:
    def test_all_architectures_present(self):
        expected = {'resnet', 'vgg', 'densenet'}
        assert set(ARCHITECTURES.keys()) == expected
    
    def test_architecture_has_required_fields(self):
        for arch in ARCHITECTURES.values():
            assert 'name' in arch
            assert 'image' in arch


class TestTrainingTypes:
    def test_all_training_types_present(self):
        expected = {'initial', 'finetuned'}
        assert set(TRAINING_TYPES.keys()) == expected
    
    def test_training_type_has_required_fields(self):
        for tt in TRAINING_TYPES.values():
            assert 'name' in tt
            assert 'folder' in tt


class TestModelConfig:
    def test_all_combinations_exist(self):
        for modality in MODALITIES:
            for arch in ARCHITECTURES:
                for tt in TRAINING_TYPES:
                    key = (modality, arch, tt)
                    assert key in MODEL_CONFIG, f"Missing config for {key}"
    
    def test_config_has_required_fields(self):
        for key, config in MODEL_CONFIG.items():
            assert 'path' in config, f"Missing path for {key}"
            assert 'threshold' in config, f"Missing threshold for {key}"
            assert 'preprocess' in config, f"Missing preprocess for {key}"
    
    def test_thresholds_are_valid(self):
        for key, config in MODEL_CONFIG.items():
            threshold = config['threshold']
            assert 0 < threshold < 1, f"Invalid threshold {threshold} for {key}"
    
    def test_preprocess_values_are_valid(self):
        valid = {'resnet', 'vgg', 'densenet'}
        for key, config in MODEL_CONFIG.items():
            assert config['preprocess'] in valid


class TestGetModelConfig:
    def test_valid_config(self):
        config = get_model_config('mammography', 'resnet', 'initial')
        assert 'path' in config
        assert 'threshold' in config
    
    def test_invalid_config_raises(self):
        with pytest.raises(ValueError):
            get_model_config('invalid', 'resnet', 'initial')


class TestGetImagePaths:
    def test_returns_all_paths(self):
        paths = get_image_paths('mammography', 'resnet', 'initial')
        assert 'confusion_matrix' in paths
        assert 'learning' in paths
        assert 'roc_curve' in paths
    
    def test_path_format(self):
        paths = get_image_paths('ultrasound', 'vgg', 'finetuned')
        assert 'ultrasound' in paths['confusion_matrix']
        assert 'fine_tune' in paths['confusion_matrix']


class TestGetDisplayInfo:
    def test_returns_all_fields(self):
        info = get_display_info('histopathology', 'densenet', 'finetuned')
        assert 'modality_name' in info
        assert 'arch_name' in info
        assert 'training_name' in info
        assert 'arch_image' in info
        assert 'title' in info
    
    def test_display_names(self):
        info = get_display_info('mammography', 'resnet', 'initial')
        assert info['modality_name'] == 'Mammography'
        assert info['arch_name'] == 'ResNet50'
        assert info['training_name'] == 'Initial Unfrozen'
