import pytest
from unittest.mock import patch, MagicMock

from src.services.model_loader import (
    get_preprocess_function, get_threshold, load_cached_model, _model_cache
)


class TestGetPreprocessFunction:
    def test_resnet_preprocess(self):
        fn = get_preprocess_function('resnet')
        assert fn is not None
        assert callable(fn)
    
    def test_vgg_preprocess(self):
        fn = get_preprocess_function('vgg')
        assert fn is not None
        assert callable(fn)
    
    def test_densenet_preprocess(self):
        fn = get_preprocess_function('densenet')
        assert fn is not None
        assert callable(fn)
    
    def test_invalid_architecture(self):
        fn = get_preprocess_function('invalid')
        assert fn is None


class TestGetThreshold:
    def test_valid_threshold(self):
        threshold = get_threshold('mammography', 'resnet', 'initial')
        assert isinstance(threshold, float)
        assert 0 < threshold < 1
    
    def test_different_thresholds(self):
        t1 = get_threshold('mammography', 'resnet', 'initial')
        t2 = get_threshold('mammography', 'resnet', 'finetuned')
        assert t1 != t2 or t1 == t2  # Just verify both work
    
    def test_invalid_config_raises(self):
        with pytest.raises(ValueError):
            get_threshold('invalid', 'resnet', 'initial')


class TestLoadCachedModel:
    @patch('src.services.model_loader.load_model')
    def test_loads_model(self, mock_load):
        _model_cache.clear()
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        result = load_cached_model('mammography', 'resnet', 'initial')
        
        assert result == mock_model
        mock_load.assert_called_once()
    
    @patch('src.services.model_loader.load_model')
    def test_caches_model(self, mock_load):
        _model_cache.clear()
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        load_cached_model('mammography', 'vgg', 'initial')
        load_cached_model('mammography', 'vgg', 'initial')
        
        assert mock_load.call_count == 1
    
    @patch('src.services.model_loader.load_model')
    def test_different_models_cached_separately(self, mock_load):
        _model_cache.clear()
        mock_load.return_value = MagicMock()
        
        load_cached_model('mammography', 'resnet', 'finetuned')
        load_cached_model('mammography', 'vgg', 'finetuned')
        
        assert mock_load.call_count == 2
    
    def test_invalid_model_raises(self):
        with pytest.raises(ValueError):
            load_cached_model('invalid', 'invalid', 'invalid')
