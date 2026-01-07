import pytest
from unittest.mock import patch, MagicMock
from app import create_app


@pytest.fixture
def app():
    app = create_app()
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


class TestMainRoutes:
    def test_index(self, client):
        response = client.get('/')
        assert response.status_code == 200
    
    def test_index_alias(self, client):
        response = client.get('/index')
        assert response.status_code == 200
    
    def test_about(self, client):
        response = client.get('/about')
        assert response.status_code == 200
    
    def test_notebook(self, client):
        response = client.get('/notebook')
        assert response.status_code == 200


class TestModalityRoutes:
    def test_mammography_page(self, client):
        response = client.get('/mammography')
        assert response.status_code == 200
    
    def test_ultrasound_page(self, client):
        response = client.get('/ultrasound')
        assert response.status_code == 200
    
    def test_histopathology_page(self, client):
        response = client.get('/histopathology')
        assert response.status_code == 200


class TestPredictionPageRoutes:
    def test_prediction_page_loads(self, client):
        response = client.get('/predict/mammography/resnet/initial')
        assert response.status_code == 200
    
    def test_all_modality_arch_combinations(self, client):
        modalities = ['mammography', 'ultrasound', 'histopathology']
        architectures = ['resnet', 'vgg', 'densenet']
        training_types = ['initial', 'finetuned']
        
        for mod in modalities:
            for arch in architectures:
                for tt in training_types:
                    response = client.get(f'/predict/{mod}/{arch}/{tt}')
                    assert response.status_code == 200, f"Failed for {mod}/{arch}/{tt}"


class TestComparisonPageRoutes:
    def test_comparison_page_loads(self, client):
        response = client.get('/compare/mammography/initial')
        assert response.status_code == 200
    
    def test_all_comparison_combinations(self, client):
        modalities = ['mammography', 'ultrasound', 'histopathology']
        training_types = ['initial', 'finetuned']
        
        for mod in modalities:
            for tt in training_types:
                response = client.get(f'/compare/{mod}/{tt}')
                assert response.status_code == 200, f"Failed for {mod}/{tt}"


class TestRedirectRoutes:
    def test_old_mammography_resnet_redirects(self, client):
        response = client.get('/resnet_mammography_initial_unfrozen')
        assert response.status_code == 301
    
    def test_old_compare_redirects(self, client):
        response = client.get('/mammography_initial_compare')
        assert response.status_code == 301


class TestPredictionPost:
    @patch('src.routes.prediction.predict_single')
    @patch('src.routes.prediction.save_temp_image')
    @patch('src.routes.prediction.cleanup_image')
    def test_prediction_post_success(self, mock_cleanup, mock_save, mock_predict, client):
        mock_save.return_value = '/tmp/test.png'
        mock_predict.return_value = {
            'label': 'Tumor Detected',
            'confidence': 0.85,
            'probability': 0.85,
            'threshold': 0.5,
            'display': 'Tumor Detected (85.00%)'
        }
        
        from io import BytesIO
        data = {'file': (BytesIO(b'fake image'), 'test.png')}
        
        response = client.post(
            '/predict/mammography/resnet/initial',
            data=data,
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 200
