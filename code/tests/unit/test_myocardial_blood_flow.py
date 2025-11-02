"""
Unit tests for myocardial_blood_flow.py
"""
import numpy as np
import pytest
from unittest.mock import patch
from myocardial_blood_flow import MyocardialBloodFlow


class TestMyocardialBloodFlow:
    """Test cases for the MyocardialBloodFlow class."""
    
    def test_compute(self, sample_aif, sample_myocardium_data, sample_tissue_impulse_response):
        """Test MyocardialBloodFlow compute method."""
        mbf = MyocardialBloodFlow(aif=sample_aif, myo=sample_myocardium_data)
        
        assert mbf.aif is not None
        assert mbf.myo is not None
        assert mbf.mbf is None      
        
        assert np.array_equal(mbf.aif, sample_aif)
        assert np.array_equal(mbf.myo, sample_myocardium_data)
        
        mbf_values = mbf.compute()
        assert mbf_values is not None
        assert isinstance(mbf_values, np.ndarray)
        assert len(mbf_values) == sample_myocardium_data.shape[1]
        
        # MBF values should be non-negative and reasonable
        assert np.all(mbf_values >= 0), "All MBF values should be non-negative"
        assert np.all(np.isfinite(mbf_values)), "All MBF values should be finite"
        
        # Verify mbf attribute is set after compute
        assert mbf.mbf is not None
        np.testing.assert_array_equal(mbf_values, mbf.mbf)
    
        # based on Quinaglia_2019: the MBF values is equal to the aplitude of impulse response function
        # So, the maximum valuse of sample_tissue_impulse_response should be equal to the MBF values
        Theoretical_MBF_values = np.max(sample_tissue_impulse_response)
        print(f"Theoretical_MBF_values: {Theoretical_MBF_values}")
        
        np.testing.assert_allclose(mbf_values, Theoretical_MBF_values, atol=10e-6)