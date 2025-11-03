"""
Tests for compute_quantity.py
"""
import numpy as np
import pytest
from compute_quantity import ComputeQuantity


class TestComputeQuantity:
    """Test cases for the ComputeQuantity class."""
    
    def test_init(self, sample_frames, sample_blood_pool_mask, sample_myocardium_mask):
        """Test ComputeQuantity initialization."""
        compute_quantity = ComputeQuantity(
            frames=sample_frames,
            blood_pool_mask=sample_blood_pool_mask,
            myo_mask=sample_myocardium_mask
        )
        assert compute_quantity.frames is not None
        assert compute_quantity.blood_pool_mask is not None
        assert compute_quantity.myo_mask is not None
        assert np.array_equal(compute_quantity.frames, sample_frames)
        assert np.array_equal(compute_quantity.blood_pool_mask, sample_blood_pool_mask)
        assert np.array_equal(compute_quantity.myo_mask, sample_myocardium_mask)
    
    
    def test_extract_pixel_time_series_shape(self, sample_frames, sample_blood_pool_mask):
        """Test _extract_pixel_time_series returns correct shape."""
        compute_quantity = ComputeQuantity(
            frames=sample_frames,
            blood_pool_mask=sample_blood_pool_mask,
            myo_mask=sample_blood_pool_mask
        )
        
        pixel_coords, time_series = compute_quantity._extract_pixel_time_series(sample_blood_pool_mask)
        
        assert len(pixel_coords) > 0
        assert isinstance(time_series, np.ndarray)
        assert time_series.ndim == 2
        assert time_series.shape[0] == len(pixel_coords)
        assert time_series.shape[1] == sample_frames.shape[0]
    
    
    def test_extract_pixel_time_series_coordinates_match_mask(self, sample_frames, sample_blood_pool_mask):
        """Test that pixel coordinates match mask pixels."""
        compute_quantity = ComputeQuantity(
            frames=sample_frames,
            blood_pool_mask=sample_blood_pool_mask,
            myo_mask=sample_blood_pool_mask
        )
        
        pixel_coords, time_series = compute_quantity._extract_pixel_time_series(sample_blood_pool_mask)
        
        # Verify that all coordinates correspond to mask pixels
        y_coords, x_coords = np.where(sample_blood_pool_mask == 1)
        expected_coords = set(zip(y_coords, x_coords))
        actual_coords = set(pixel_coords)
        assert expected_coords == actual_coords
    
    
    def test_blood_pool_time_series(self, sample_frames, sample_blood_pool_mask, sample_myocardium_mask):
        """Test if the values of blood_pool_time_series method are equal to the values of sample_blood_pool_data."""
        compute_quantity = ComputeQuantity(
            frames=sample_frames,
            blood_pool_mask=sample_blood_pool_mask,
            myo_mask=sample_myocardium_mask
        )
        
        pixel_coords, time_series = compute_quantity.blood_pool_time_series()
        
        # Extract the data of th 1st pixel of blood pool data from the frames
        y_coords, x_coords = np.where(sample_blood_pool_mask == 1)
        y_coord = y_coords[0]
        x_coord = x_coords[0]
        sample_blood_pool_data = sample_frames[:, y_coord, x_coord]
        
        np.testing.assert_array_equal(time_series[0], sample_blood_pool_data)
    
    
    def test_myocardium_time_series(self, sample_frames, sample_blood_pool_mask, sample_myocardium_mask):
        """Test if the values of myocardium_time_series method are equal to the values of sample_myocardium_data."""
        compute_quantity = ComputeQuantity(
            frames=sample_frames,
            blood_pool_mask=sample_blood_pool_mask,
            myo_mask=sample_myocardium_mask
        )
        
        pixel_coords, time_series = compute_quantity.myocardium_time_series()
        
        # Extract the data of the 1st pixel of myocardium data from the frames
        y_coords, x_coords = np.where(sample_myocardium_mask == 1)
        y_coord = y_coords[0]
        x_coord = x_coords[0]
        sample_myocardium_data = sample_frames[:, y_coord, x_coord]
        
        np.testing.assert_array_equal(time_series[0], sample_myocardium_data)
    
    
    def test_arterial_input_function(self, sample_frames, sample_blood_pool_mask, sample_myocardium_mask):
        """Test arterial_input_function computation."""
        compute_quantity = ComputeQuantity(
            frames=sample_frames,
            blood_pool_mask=sample_blood_pool_mask,
            myo_mask=sample_myocardium_mask
        )
        
        aif = compute_quantity.arterial_input_function()
        
        
        # Extract the data of th 1st pixel of blood pool data from the frames
        y_coords, x_coords = np.where(sample_blood_pool_mask == 1)
        y_coord = y_coords[0]
        x_coord = x_coords[0]
        sample_blood_pool_data = sample_frames[:, y_coord, x_coord]
        
        #AIF should be equal to the values for one of pixel in blood pool over the time
        np.testing.assert_array_equal(aif, sample_blood_pool_data)
    
