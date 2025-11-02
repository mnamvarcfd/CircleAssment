"""
Unit tests for myocardial_blood_flow.py
"""
import numpy as np
import pytest
from unittest.mock import patch
from myocardial_blood_flow import MyocardialBloodFlow


class TestMyocardialBloodFlow:
    """Test cases for the MyocardialBloodFlow class."""
    
    def test_fermi_function_fitting_same_as_given_parameters(self, sample_aif, sample_myocardium_data, sample_myocardium_mask):
        """Test MyocardialBloodFlow fermi_function_fitting method with given parameters."""
        myocardial_blood_flow = MyocardialBloodFlow(aif=sample_aif, myo=sample_myocardium_data, myo_mask=sample_myocardium_mask)

        # Extract a single pixel's time series where mask is 1
        y_coords, x_coords = np.where(sample_myocardium_mask == 1)
        test_pixel_y, test_pixel_x = y_coords[0], x_coords[0]  # Use first valid pixel
        single_pixel_curve = sample_myocardium_data[:, test_pixel_y, test_pixel_x]

        mbf_value = myocardial_blood_flow.fermi_function_fitting(single_pixel_curve)
        
        assert mbf_value is not None
        assert isinstance(mbf_value, float)
        assert np.isclose(mbf_value, 1.0, atol=10e-6)
        

    def test_fermi_function_fitting_with_different_parameters(self, sample_aif, sample_myocardium_data, sample_myocardium_mask):
        """Test MyocardialBloodFlow fermi_function_fitting method with different parameters."""
        myocardial_blood_flow = MyocardialBloodFlow(aif=sample_aif, myo=sample_myocardium_data, myo_mask=sample_myocardium_mask)

        # Extract a single pixel's time series where mask is 1
        y_coords, x_coords = np.where(sample_myocardium_mask == 1)
        test_pixel_y, test_pixel_x = y_coords[0], x_coords[0]  # Use first valid pixel
        single_pixel_curve = sample_myocardium_data[:, test_pixel_y, test_pixel_x]

        mbf_value = myocardial_blood_flow.fermi_function_fitting(single_pixel_curve, F_init=2.0, tau_0_init=30.0, k_init=0.2)
        
        assert mbf_value is not None
        assert isinstance(mbf_value, float)
        assert np.isclose(mbf_value, 1.0, atol=10e-6)
        
        
    def test_compute(self, sample_aif, sample_myocardium_data, sample_tissue_impulse_response, sample_myocardium_mask):
        """Test MyocardialBloodFlow compute method."""
        myocardial_blood_flow = MyocardialBloodFlow(aif=sample_aif, myo=sample_myocardium_data, myo_mask=sample_myocardium_mask)
     
        mbf_values = myocardial_blood_flow.compute()   
        
        assert myocardial_blood_flow.aif.shape == sample_aif.shape
        assert myocardial_blood_flow.myo.shape == sample_myocardium_data.shape

        assert mbf_values is not None
        assert isinstance(mbf_values, np.ndarray)
        assert mbf_values.shape == (sample_myocardium_data.shape[1], sample_myocardium_data.shape[2])
        
        assert np.array_equal(myocardial_blood_flow.aif, sample_aif)
        assert np.array_equal(myocardial_blood_flow.myo, sample_myocardium_data)
     
        # Verify mbf attribute is set after compute
        assert myocardial_blood_flow.mbf is not None

        # # based on Quinaglia_2019: the MBF values is equal to the aplitude of impulse response function
        # # So, the MBF values should be close to the A (Amplitude) of impulse response function used to 
        # generate these values.
        y_coords, x_coords = np.where(sample_myocardium_mask == 1)
        myocardium_pixel = zip(y_coords, x_coords)
        for y, x in myocardium_pixel:
            np.testing.assert_allclose(mbf_values[y, x], 1.0, atol=10e-6)
    
        #This is only for presentation =================================================
        from save_data_manager import SaveDataManager
        manager = SaveDataManager(results_dir='F:/18_Circle/code/tests/test_outputs')
        print(f"----test_directory: F:/18_Circle/code/tests/test_outputs")

        # Extract MBF values only for myocardium pixels (where mask == 1)
        myocardium_mbf_values = mbf_values[sample_myocardium_mask == 1]

        manager.save_image(
            myocardium_mbf_values,
            sample_myocardium_mask,
            Value_title="MBF",
            output_filename="mbf_map_analytical.png"
        )
        #==============================================================================
    