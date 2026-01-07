import os
import pytest
import numpy as np
from io import BytesIO
from PIL import Image
from unittest.mock import MagicMock

from src.utils.image_processing import (
    allowed_file, load_and_preprocess, save_temp_image, cleanup_image
)


class TestAllowedFile:
    def test_png_allowed(self):
        assert allowed_file('test.png') is True
    
    def test_jpg_allowed(self):
        assert allowed_file('test.jpg') is True
    
    def test_jpeg_allowed(self):
        assert allowed_file('test.jpeg') is True
    
    def test_uppercase_allowed(self):
        assert allowed_file('test.PNG') is True
        assert allowed_file('test.JPG') is True
    
    def test_gif_not_allowed(self):
        assert allowed_file('test.gif') is False
    
    def test_no_extension(self):
        assert allowed_file('testfile') is False
    
    def test_empty_string(self):
        assert allowed_file('') is False


class TestLoadAndPreprocess:
    @pytest.fixture
    def temp_image(self, tmp_path):
        img = Image.new('RGB', (300, 300), color='red')
        path = tmp_path / 'test.png'
        img.save(path)
        return str(path)
    
    def test_output_shape(self, temp_image):
        result = load_and_preprocess(temp_image)
        assert result.shape == (1, 224, 224, 3)
    
    def test_output_type(self, temp_image):
        result = load_and_preprocess(temp_image)
        assert isinstance(result, np.ndarray)
    
    def test_batch_dimension(self, temp_image):
        result = load_and_preprocess(temp_image)
        assert result.ndim == 4


class TestSaveTempImage:
    @pytest.fixture
    def mock_file(self, tmp_path):
        mock = MagicMock()
        mock.filename = 'test_upload.png'
        mock.save = MagicMock()
        return mock
    
    def test_returns_path(self, mock_file):
        path = save_temp_image(mock_file)
        assert 'test_upload.png' in path
    
    def test_calls_save(self, mock_file):
        save_temp_image(mock_file)
        mock_file.save.assert_called_once()


class TestCleanupImage:
    def test_removes_existing_file(self, tmp_path):
        path = tmp_path / 'to_delete.png'
        path.touch()
        assert path.exists()
        cleanup_image(str(path))
        assert not path.exists()
    
    def test_handles_nonexistent_file(self, tmp_path):
        path = tmp_path / 'nonexistent.png'
        cleanup_image(str(path))  # Should not raise
