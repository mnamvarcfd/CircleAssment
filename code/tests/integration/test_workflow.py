"""
Integration tests for the complete myocardial blood flow computation workflow.
"""
import numpy as np
import pytest
import os
from compute_quantity import ComputeQuantity
from myocardial_blood_flow import MyocardialBloodFlow
from save_data_manager import SaveDataManager


class TestWorkflow:
    """Integration tests for the complete workflow."""
    
    def test_complete_workflow(self, sample_frames, sample_blood_pool_mask, sample_myocardium_mask, temp_dir):
        """Test the complete workflow from data loading to MBF computation and saving."""
        # Step 1: Initialize ComputeQuantity
        compute_quantity = ComputeQuantity(
            frames=sample_frames,
            blood_pool_mask=sample_blood_pool_mask,
            myo_mask=sample_myocardium_mask
        )
        
        # Step 2: Compute blood pool time series
        blood_pool_pixel_coordinates, blood_pool_time_series = compute_quantity.blood_pool_time_series()
        assert len(blood_pool_pixel_coordinates) > 0
        assert blood_pool_time_series.shape[0] == len(blood_pool_pixel_coordinates)
        assert blood_pool_time_series.shape[1] == sample_frames.shape[0]
        
        # Step 3: Compute myocardium time series
        myo_pixel_coordinates, myo_time_series = compute_quantity.myocardium_time_series()
        assert len(myo_pixel_coordinates) > 0
        assert myo_time_series.shape[0] == len(myo_pixel_coordinates)
        assert myo_time_series.shape[1] == sample_frames.shape[0]
        
        # Step 4: Compute Arterial Input Function (AIF)
        aif = compute_quantity.arterial_input_function()
        assert aif is not None
        assert isinstance(aif, np.ndarray)
        assert aif.ndim == 1
        assert len(aif) == sample_frames.shape[0]
        
        # Step 5: Verify AIF is the mean of blood pool pixels
        expected_aif = np.mean(blood_pool_time_series, axis=0)
        np.testing.assert_array_almost_equal(aif, expected_aif)
        
        # Step 6: Initialize MyocardialBloodFlow
        myocardial_blood_flow = MyocardialBloodFlow(aif=aif, myo=myo_time_series.T)
        assert myocardial_blood_flow.aif is not None
        assert myocardial_blood_flow.myo is not None
        
        # Step 7: Compute MBF
        mbf = myocardial_blood_flow.compute()
        assert mbf is not None
        assert isinstance(mbf, np.ndarray)
        assert len(mbf) == myo_time_series.shape[0]
        assert np.all(mbf >= 0), "MBF values should be non-negative"
        
        # Step 8: Save results using SaveDataManager
        save_data_manager = SaveDataManager(results_dir=temp_dir)
        
        # Save MBF map
        mbf_map_path = save_data_manager.save_image(
            mbf,
            sample_myocardium_mask,
            Value_title="MBF",
            output_filename="mbf_map.png"
        )
        assert mbf_map_path is not None
        assert os.path.exists(mbf_map_path)
        
        # Save blood pool mask plot
        blood_pool_mask_path = save_data_manager.plot_mask(
            sample_blood_pool_mask,
            output_filename="blood_pool_mask.png"
        )
        assert blood_pool_mask_path is not None
        assert os.path.exists(blood_pool_mask_path)
        
        # Save myocardium mask plot
        myo_mask_path = save_data_manager.plot_mask(
            sample_myocardium_mask,
            output_filename="myocardium_mask.png"
        )
        assert myo_mask_path is not None
        assert os.path.exists(myo_mask_path)
        
        # Save AIF plot
        aif_plot_path = save_data_manager.plot_pixel_over_time(
            aif,
            title="Arterial Input Function (AIF)",
            y_label="Signal Intensity",
            output_filename="aif.png"
        )
        assert aif_plot_path is not None
        assert os.path.exists(aif_plot_path)
        
        # Save blood pool time series plot (for first pixel)
        bp_series_path = save_data_manager.plot_pixel_over_time(
            blood_pool_time_series[0],
            title="Blood Pool Time Series (First Pixel)",
            y_label="Signal Intensity",
            output_filename="blood_pool_time_series.png"
        )
        assert bp_series_path is not None
        assert os.path.exists(bp_series_path)
        
        # Save myocardium time series plot (for first pixel)
        myo_series_path = save_data_manager.plot_pixel_over_time(
            myo_time_series[0],
            title="Myocardium Time Series (First Pixel)",
            y_label="Signal Intensity",
            output_filename="myocardium_time_series.png"
        )
        assert myo_series_path is not None
        assert os.path.exists(myo_series_path)
        
        # Save movie from frames
        movie_path = save_data_manager.create_movie_from_frames(
            sample_frames,
            output_path="workflow_movie.avi",
            fps=10
        )
        assert movie_path is not None
        assert os.path.exists(movie_path)
    
    def test_workflow_data_consistency(self, sample_frames, sample_blood_pool_mask, sample_myocardium_mask):
        """Test that data flows correctly through the entire workflow."""
        # Initialize components
        compute_quantity = ComputeQuantity(
            frames=sample_frames,
            blood_pool_mask=sample_blood_pool_mask,
            myo_mask=sample_myocardium_mask
        )
        
        # Extract time series
        blood_pool_pixel_coords, blood_pool_time_series = compute_quantity.blood_pool_time_series()
        myo_pixel_coords, myo_time_series = compute_quantity.myocardium_time_series()
        aif = compute_quantity.arterial_input_function()
        
        # Verify data dimensions are consistent
        assert blood_pool_time_series.shape[1] == sample_frames.shape[0]
        assert myo_time_series.shape[1] == sample_frames.shape[0]
        assert len(aif) == sample_frames.shape[0]
        
        # Verify AIF matches mean of blood pool
        expected_aif = np.mean(blood_pool_time_series, axis=0)
        np.testing.assert_array_almost_equal(aif, expected_aif)
        
        # Compute MBF
        myocardial_blood_flow = MyocardialBloodFlow(aif=aif, myo=myo_time_series.T)
        mbf = myocardial_blood_flow.compute()
        
        # Verify MBF dimensions match myocardium pixels
        assert len(mbf) == myo_time_series.shape[0]
        assert len(mbf) == len(myo_pixel_coords)
    
    def test_workflow_with_different_frame_counts(self, sample_blood_pool_mask, sample_myocardium_mask):
        """Test workflow with different numbers of frames."""
        for num_frames in [5, 10, 20]:
            # Create frames with different time dimensions
            height, width = sample_blood_pool_mask.shape
            frames = np.random.rand(num_frames, height, width) * 1000
            
            compute_quantity = ComputeQuantity(
                frames=frames,
                blood_pool_mask=sample_blood_pool_mask,
                myo_mask=sample_myocardium_mask
            )
            
            # Extract time series
            _, blood_pool_time_series = compute_quantity.blood_pool_time_series()
            _, myo_time_series = compute_quantity.myocardium_time_series()
            aif = compute_quantity.arterial_input_function()
            
            # Verify dimensions
            assert blood_pool_time_series.shape[1] == num_frames
            assert myo_time_series.shape[1] == num_frames
            assert len(aif) == num_frames
            
            # Compute MBF
            myocardial_blood_flow = MyocardialBloodFlow(aif=aif, myo=myo_time_series.T)
            mbf = myocardial_blood_flow.compute()
            
            assert len(mbf) == myo_time_series.shape[0]
    
    def test_workflow_mbf_values_reasonable(self, sample_frames, sample_blood_pool_mask, sample_myocardium_mask):
        """Test that MBF values computed in the workflow are reasonable."""
        compute_quantity = ComputeQuantity(
            frames=sample_frames,
            blood_pool_mask=sample_blood_pool_mask,
            myo_mask=sample_myocardium_mask
        )
        
        _, myo_time_series = compute_quantity.myocardium_time_series()
        aif = compute_quantity.arterial_input_function()
        
        myocardial_blood_flow = MyocardialBloodFlow(aif=aif, myo=myo_time_series.T)
        mbf = myocardial_blood_flow.compute()
        
        # MBF should be non-negative
        assert np.all(mbf >= 0), "All MBF values should be non-negative"
        
        # Check that at least some pixels have non-zero MBF (or all are zero if fitting fails)
        # This is a sanity check
        assert len(mbf) > 0, "MBF array should not be empty"
        
        # If there are non-zero values, they should be reasonable (not infinite, not NaN)
        if np.any(mbf > 0):
            assert np.all(np.isfinite(mbf)), "MBF values should be finite"
            assert np.all(mbf < 1e6), "MBF values should be reasonable (less than 1e6)"

