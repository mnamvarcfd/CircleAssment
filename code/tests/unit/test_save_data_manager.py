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
    
    def test_plot_mask(self, temp_dir, sample_mask):
        """Test that plot_mask saves a blood pool mask plot correctly."""
        manager = SaveDataManager(results_dir=temp_dir)
        output_path = manager.plot_mask(sample_mask, output_filename="sample_mask.png")
        
        assert output_path is not None, "plot_mask should return the output file path"
        expected_path = os.path.join(temp_dir, "sample_mask.png")
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
        
      
    def test_save_image(self, temp_dir, sample_mask):
        """Test that save_image saves an image map correctly."""
        manager = SaveDataManager(results_dir=temp_dir)
        # Create pixel values matching the number of mask pixels
        num_pixels = np.sum(sample_mask == 1)
        pixel_values = np.ones(num_pixels)
        
        output_path = manager.save_image(
            pixel_values,
            sample_mask,
            Value_title="Test Values",
            output_filename="sample_image.png"
        )
        
        assert output_path is not None
        expected_path = os.path.join(temp_dir, "sample_image.png")
        assert output_path == expected_path
        assert os.path.exists(output_path)
    
    
    def test_create_movie_from_frames(self, temp_dir, sample_frames):
        """Test that create_movie_from_frames creates a video file."""
        manager = SaveDataManager(results_dir=temp_dir)
        
        output_path = manager.create_movie_from_frames(
            sample_frames,
            output_path="sample_movie.avi",
            fps=1
        )
        
        assert output_path is not None
        expected_path = os.path.join(temp_dir, "sample_movie.avi")
        assert output_path == expected_path
    
