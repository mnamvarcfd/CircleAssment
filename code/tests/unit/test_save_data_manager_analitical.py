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
        
       
    def test_plot_pixel_over_time_blood_pool(self, temp_dir, sample_frames, sample_blood_pool_mask):
        """Test that plot_pixel_over_time creates a plot."""
        manager = SaveDataManager(results_dir=temp_dir)
        
        
        # Extract the data of th 1st pixel of blood pool data from the frames
        y_coords, x_coords = np.where(sample_blood_pool_mask == 1)
        y_coord = y_coords[0]
        x_coord = x_coords[0]
        sample_blood_pool_data = sample_frames[:, y_coord, x_coord]
        
        output_path = manager.plot_pixel_over_time(
            sample_blood_pool_data,
            title="Test Blood Pool Time Series",
            y_label="Signal Intensity",
            output_filename="test_blood_pool_time_series.png"
        )
        
        assert output_path is not None
        expected_path = os.path.join(temp_dir, "test_blood_pool_time_series.png")
        assert output_path == expected_path
        assert os.path.exists(output_path)
        
        
    def test_plot_pixel_over_time_AIF(self, temp_dir, sample_aif):
        """Test that plot_pixel_over_time creates a plot for AIF."""
        manager = SaveDataManager(results_dir=temp_dir)
        
        output_path = manager.plot_pixel_over_time(
            sample_aif,
            title="Test AIF Time Series",
            y_label="Signal Intensity",
            output_filename="test_aif_time_series.png"
        )
        
        assert output_path is not None
        expected_path = os.path.join(temp_dir, "test_aif_time_series.png")
        assert output_path == expected_path
        assert os.path.exists(output_path)
        
      
    def test_plot_pixel_over_time_myocardium(self, temp_dir, sample_frames, sample_myocardium_mask):
        """Test that plot_pixel_over_time creates a plot for myocardium time series."""
        manager = SaveDataManager(results_dir=temp_dir)
        
        
        # Extract the data of th 1st pixel of myocardium data from the frames
        y_coords, x_coords = np.where(sample_myocardium_mask == 1)
        y_coord = y_coords[0]
        x_coord = x_coords[0]
        sample_myocardium_data = sample_frames[:, y_coord, x_coord]
        
        output_path = manager.plot_pixel_over_time(
            sample_myocardium_data,
            title="Test Myocardium Time Series",
            y_label="Signal Intensity",
            output_filename="test_myocardium_time_series.png"
        )
        
        assert output_path is not None
        expected_path = os.path.join(temp_dir, "test_myocardium_time_series.png")
        assert output_path == expected_path
        assert os.path.exists(output_path)
      
        
    def test_plot_pixel_over_time_tissue_impulse_response(self, temp_dir, sample_tissue_impulse_response, sample_myocardium_mask):
        """Test that plot_pixel_over_time creates a plot for tissue impulse response time series."""
        manager = SaveDataManager(results_dir=temp_dir)
        
        # Find a pixel that's actually in the myocardium mask (not all zeros)
        y_coords, x_coords = np.where(sample_myocardium_mask == 1)
        
        # Use the first pixel in the mask
        y, x = y_coords[0], x_coords[0]
        pixel_time_series = sample_tissue_impulse_response[:, y, x]
        
        output_path = manager.plot_pixel_over_time(
            pixel_time_series,
            title="Test Tissue Impulse Response Time Series",
            y_label="Signal Intensity",
            output_filename="test_tissue_impulse_response_time_series.png"
        )
        
        assert output_path is not None
        expected_path = os.path.join(temp_dir, "test_tissue_impulse_response_time_series.png")
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
    
