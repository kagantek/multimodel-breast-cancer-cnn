import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestPredictSingle:
    @patch('src.services.predictor.load_and_preprocess')
    @patch('src.services.predictor.get_threshold')
    @patch('src.services.predictor.load_cached_model')
    def test_tumor_detected(self, mock_model, mock_threshold, mock_preprocess):
        from src.services.predictor import predict_single
        
        mock_model.return_value.predict.return_value = np.array([[0.8]])
        mock_threshold.return_value = 0.5
        mock_preprocess.return_value = np.zeros((1, 224, 224, 3))
        
        result = predict_single('test.png', 'mammography', 'resnet', 'initial')
        
        assert result['label'] == 'Tumor Detected'
        assert result['probability'] == 0.8
        assert result['threshold'] == 0.5
    
    @patch('src.services.predictor.load_and_preprocess')
    @patch('src.services.predictor.get_threshold')
    @patch('src.services.predictor.load_cached_model')
    def test_no_tumor_detected(self, mock_model, mock_threshold, mock_preprocess):
        from src.services.predictor import predict_single
        
        mock_model.return_value.predict.return_value = np.array([[0.3]])
        mock_threshold.return_value = 0.5
        mock_preprocess.return_value = np.zeros((1, 224, 224, 3))
        
        result = predict_single('test.png', 'mammography', 'resnet', 'initial')
        
        assert result['label'] == 'No Tumor Detected'
        assert result['probability'] == 0.3
    
    @patch('src.services.predictor.load_and_preprocess')
    @patch('src.services.predictor.get_threshold')
    @patch('src.services.predictor.load_cached_model')
    def test_confidence_calculation_tumor(self, mock_model, mock_threshold, mock_preprocess):
        from src.services.predictor import predict_single
        
        mock_model.return_value.predict.return_value = np.array([[0.9]])
        mock_threshold.return_value = 0.5
        mock_preprocess.return_value = np.zeros((1, 224, 224, 3))
        
        result = predict_single('test.png', 'mammography', 'resnet', 'initial')
        
        assert result['confidence'] == 0.9
    
    @patch('src.services.predictor.load_and_preprocess')
    @patch('src.services.predictor.get_threshold')
    @patch('src.services.predictor.load_cached_model')
    def test_confidence_calculation_no_tumor(self, mock_model, mock_threshold, mock_preprocess):
        from src.services.predictor import predict_single
        
        mock_model.return_value.predict.return_value = np.array([[0.2]])
        mock_threshold.return_value = 0.5
        mock_preprocess.return_value = np.zeros((1, 224, 224, 3))
        
        result = predict_single('test.png', 'mammography', 'resnet', 'initial')
        
        assert result['confidence'] == 0.8
    
    @patch('src.services.predictor.load_and_preprocess')
    @patch('src.services.predictor.get_threshold')
    @patch('src.services.predictor.load_cached_model')
    def test_result_has_display(self, mock_model, mock_threshold, mock_preprocess):
        from src.services.predictor import predict_single
        
        mock_model.return_value.predict.return_value = np.array([[0.7]])
        mock_threshold.return_value = 0.5
        mock_preprocess.return_value = np.zeros((1, 224, 224, 3))
        
        result = predict_single('test.png', 'mammography', 'resnet', 'initial')
        
        assert 'display' in result
        assert 'Threshold' in result['display']


class TestPredictComparison:
    @patch('src.services.predictor.load_and_preprocess')
    @patch('src.services.predictor.get_threshold')
    @patch('src.services.predictor.load_cached_model')
    def test_returns_all_architectures(self, mock_model, mock_threshold, mock_preprocess):
        from src.services.predictor import predict_comparison
        
        mock_model.return_value.predict.return_value = np.array([[0.6]])
        mock_threshold.return_value = 0.5
        mock_preprocess.return_value = np.zeros((1, 224, 224, 3))
        
        result = predict_comparison('test.png', 'mammography', 'initial')
        
        assert 'ResNet50' in result
        assert 'VGG16' in result
        assert 'DenseNet121' in result
    
    @patch('src.services.predictor.load_and_preprocess')
    @patch('src.services.predictor.get_threshold')
    @patch('src.services.predictor.load_cached_model')
    def test_each_result_has_required_fields(self, mock_model, mock_threshold, mock_preprocess):
        from src.services.predictor import predict_comparison
        
        mock_model.return_value.predict.return_value = np.array([[0.6]])
        mock_threshold.return_value = 0.5
        mock_preprocess.return_value = np.zeros((1, 224, 224, 3))
        
        result = predict_comparison('test.png', 'mammography', 'initial')
        
        for arch_result in result.values():
            assert 'label' in arch_result
            assert 'conf' in arch_result
            assert 'tau' in arch_result
