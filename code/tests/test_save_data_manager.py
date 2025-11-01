"""
Tests for save_data_manager.py
"""
import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, mock_open
from save_data_manager import SaveDataManager


class TestSaveDataManager:
    """Test cases for the SaveDataManager class."""
    
    def test_plot_mask(self, sample_blood_pool_mask, temp_dir):
        """Test that plot_mask saves a blood pool mask plot correctly."""
        manager = SaveDataManager(results_dir=temp_dir)
        output_path = manager.plot_mask(sample_blood_pool_mask, output_filename="test_mask.png")
        
        assert output_path is not None, "plot_mask should return the output file path"
        expected_path = os.path.join(temp_dir, "test_mask.png")
        assert output_path == expected_path
        assert os.path.exists(output_path)
    
    
    def test_plot_pixel_over_time(self, temp_dir):
        """Test that plot_pixel_over_time creates a plot."""
        manager = SaveDataManager(results_dir=temp_dir)
        series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        output_path = manager.plot_pixel_over_time(
            series,
            title="Test Series",
            y_label="Value",
            output_filename="test_series.png"
        )
        
        assert output_path is not None
        expected_path = os.path.join(temp_dir, "test_series.png")
        assert output_path == expected_path
        assert os.path.exists(output_path)
        
        
    def test_save_image_blood_pool(self, sample_blood_pool_mask, temp_dir):
        """Test that save_image saves an image map correctly."""
        manager = SaveDataManager(results_dir=temp_dir)
        # Create pixel values matching the number of mask pixels
        num_pixels = np.sum(sample_blood_pool_mask == 1)
        pixel_values = np.ones(num_pixels)
        
        output_path = manager.save_image(
            pixel_values,
            sample_blood_pool_mask,
            Value_title="Test Values",
            output_filename="test_image_blood_pool.png"
        )
        
        assert output_path is not None
        expected_path = os.path.join(temp_dir, "test_image_blood_pool.png")
        assert output_path == expected_path
        assert os.path.exists(output_path)
    
        
    def test_save_image_myocardium(self, sample_myocardium_mask, temp_dir):
        """Test that save_image saves a myocardium image map correctly."""
        manager = SaveDataManager(results_dir=temp_dir)
        # Create pixel values matching the number of mask pixels
        num_pixels = np.sum(sample_myocardium_mask == 1)
        pixel_values = np.ones(num_pixels)
        
        output_path = manager.save_image(
            pixel_values,
            sample_myocardium_mask,
            Value_title="Test Values",
            output_filename="test_image_myocardium.png"
        )
        
        assert output_path is not None
        expected_path = os.path.join(temp_dir, "test_image_myocardium.png")
        assert output_path == expected_path
        assert os.path.exists(output_path)
    
    
    def test_create_movie_from_frames(self, sample_frames, temp_dir):
        """Test that create_movie_from_frames creates a video file."""
        manager = SaveDataManager(results_dir=temp_dir)
        
        output_path = manager.create_movie_from_frames(
            sample_frames,
            output_path="test_movie.avi",
            fps=1
        )
        
        assert output_path is not None
        expected_path = os.path.join(temp_dir, "dicom_movie.avi")
        assert output_path == expected_path
    
