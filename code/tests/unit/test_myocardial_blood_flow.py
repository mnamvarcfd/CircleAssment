"""
Unit tests for myocardial_blood_flow.py
"""
import os
import numpy as np
import pytest
import pandas as pd
from unittest.mock import patch
from scipy.signal import convolve
from myocardial_blood_flow_fixed_tau_d import MyocardialBloodFlow
from save_data_manager import SaveDataManager
from tests.test_utils import gamma_variate
    
    
@pytest.fixture
def sample_blood_pool_mask():
    """Create a sample 2D binary mask for blood pool region."""
    mask = np.zeros((1, 2), dtype=np.uint8)
    
    mask[0, 0] = 1 
    
    return mask


@pytest.fixture
def sample_myocardium_mask():
    """Create a sample 2D binary mask for myocardium region."""
    mask = np.zeros((1, 2), dtype=np.uint8)
    
    mask[0, 1] = 1  

    return mask


@pytest.fixture
def sample_frames(sample_blood_pool_mask, sample_myocardium_mask):
    """Create sample frames with realistic AIF and myocardium data."""
    frames = np.zeros((60, 1, 2), dtype=np.float64)

    # Generate AIF signal
    t = np.arange(60)
    aif_signal = gamma_variate(t, init_value=10, A=150, alpha=3.5, beta=4.5)

    # Generate myocardium signal by convolving AIF with Fermi function
    R_F = MyocardialBloodFlow.fermi_function(t, F=1.0, tau_0=20.0, k=0.1)
    myo_signal = convolve(aif_signal, R_F, mode='full')[:len(aif_signal)]

    # Put signals in appropriate regions
    for i in range(frames.shape[0]):
        frames[i, sample_blood_pool_mask == 1] = aif_signal[i]
        frames[i, sample_myocardium_mask == 1] = myo_signal[i]

    return frames


class TestMyocardialBloodFlow:
    """Test cases for the MyocardialBloodFlow class."""
    
    def test_fermi_curve_fitting_given_parameters(self, 
                                                  sample_frames, 
                                                  sample_blood_pool_mask, 
                                                  sample_myocardium_mask,
                                                  unit_test_results_dir):
        """
        Test fermi_curve_fitting method with given parameters.

        An specific AIF is uses (already generated as a fixture in conftest.py).
        A fermi function curve (already generated as a fixture in conftest.py) is
        used to generate the myocardium curve.
        So, in this case, the parameters used to generate the myocardium curve are known.

        Hence, the MBF value (returned by fermi_curve_fitting method) should be as the
        one used to generate the myocardium curve (1.0 in this case).
        """
        # Create SaveDataManager for saving results
        save_manager = SaveDataManager(results_dir=unit_test_results_dir)
        
        # Create MyocardialBloodFlow instance
        myocardial_blood_flow = MyocardialBloodFlow(
            frames=sample_frames,
            blood_pool_mask=sample_blood_pool_mask,
            myo_mask=sample_myocardium_mask
        )

        t = np.arange(60)
        known_F = 1.0
        tau_0 = 20.0
        k = 0.1

        # Generate impulse response (Fermi function)
        R_F = MyocardialBloodFlow.fermi_function(t, known_F, tau_0, k)

        # Convolve with AIF to generate myocardium time series with known parameters
        myo_time_series = convolve(myocardial_blood_flow.aif, R_F, mode='full')[:len(R_F)]

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Covariance of the parameters could not be estimated")
            mbf_value = myocardial_blood_flow.fermi_curve_fitting(
                myo_time_series,
                F_init=1.0,      # Exact match with true value
                tau_0_init=20.0, # Exact match with true value
                k_init=0.1       # Exact match with true value
            )

        assert mbf_value is not None
        assert isinstance(mbf_value, float)
        assert np.isclose(mbf_value, 1.0, atol=10e-6)
        

    def test_fermi_function_fitting_with_unknown_parameters(self, 
                                                            sample_frames, 
                                                            sample_blood_pool_mask, 
                                                            sample_myocardium_mask):
        """
        Test MyocardialBloodFlow fermi_function_fitting method with unknown parameters.
        
        An specific AIF is uses (already generated as a fixture in conftest.py).
        A fermi function curve (already generated as a fixture in conftest.py) is
        used to generate the myocardium curve.
        So, in this case, the parameters used to generate the myocardium curve are known.

        Hence, the MBF value (returned by fermi_curve_fitting method) should be as the
        one used to generate the myocardium curve (1.0 in this case).
        """
        # Create MyocardialBloodFlow instance
        myocardial_blood_flow = MyocardialBloodFlow(
            frames=sample_frames,
            blood_pool_mask=sample_blood_pool_mask,
            myo_mask=sample_myocardium_mask
        )

        t = np.arange(60)
        known_F = 1.0
        tau_0 = 20.0
        k = 0.1

        # Generate impulse response (Fermi function)
        R_F = MyocardialBloodFlow.fermi_function(t, known_F, tau_0, k)

        # Convolve with AIF to generate myocardium time series with known parameters
        myo_time_series = convolve(myocardial_blood_flow.aif, R_F, mode='full')[:len(R_F)]

        mbf_value = myocardial_blood_flow.fermi_curve_fitting(
            myo_time_series,
            F_init=0.5,      # Reasonable initial guess (half the true value)
            tau_0_init=10.0, # Reasonable initial guess (half the true value)
            k_init=0.5      # Reasonable initial guess (5x the true value, but reasonable)
        )

        assert mbf_value is not None
        assert isinstance(mbf_value, float)
        # Should converge reasonably well from reasonable initial guesses
        assert np.isclose(mbf_value, 1.0, atol=10e-6)
        
        
    def test_compute(self, 
                     sample_frames, 
                     sample_blood_pool_mask, 
                     sample_myocardium_mask):
        """Test MyocardialBloodFlow compute method."""
        
        # Create MyocardialBloodFlow instance
        myocardial_blood_flow = MyocardialBloodFlow(
            frames=sample_frames,
            blood_pool_mask=sample_blood_pool_mask,
            myo_mask=sample_myocardium_mask
        )
        
        mbf_values = myocardial_blood_flow.compute()   
        
        assert myocardial_blood_flow.aif.shape == (60,)
        assert myocardial_blood_flow.myo_pixel_time_series.shape == (1, 60)

        assert mbf_values is not None
        assert isinstance(mbf_values, np.ndarray)
        assert mbf_values.shape == (1,)
        
        # # based on Quinaglia_2019: the MBF values is equal to the aplitude of impulse response function
        # # So, the MBF values should be close to the A (Amplitude) of impulse response function used to 
        # generate these values.
        np.testing.assert_allclose(mbf_values[0], 1.0, atol=10e-6)
            
    
    