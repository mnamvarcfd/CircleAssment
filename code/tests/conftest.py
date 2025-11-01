"""
Shared pytest fixtures for testing.
"""
import os
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import tempfile
import shutil

# Add parent directory to path to allow importing src modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


num_frames = 10
image_size = 50

def gamma_variate(t, A=1.0, t0=0.0, alpha=2.0, beta=1.0):
    """Gamma-variate function: rise and decay curve"""
    f = np.zeros_like(t)
    mask = t > t0
    f[mask] = A * ((t[mask] - t0) ** alpha) * np.exp(-(t[mask] - t0) / beta)
    return f


@pytest.fixture
def sample_blood_pool_data(sample_blood_pool_mask):
    """Create sample data for blood pool time series."""
    blood_pool_mask = sample_blood_pool_mask
    
    blood_pool_data = np.zeros((num_frames, image_size, image_size), dtype=np.float64)
    
    y_coords, x_coords = np.where(blood_pool_mask == 1)
    
    time_series = gamma_variate(np.arange(num_frames), A=1000, t0=0, alpha=2, beta=1)
    
    for y, x in zip(y_coords, x_coords):
        blood_pool_data[:, y, x] = time_series
        
    return blood_pool_data


@pytest.fixture
def sample_myocardium_data(sample_myocardium_mask):
    """Create sample data for myocardium time series."""
    myocardium_mask = sample_myocardium_mask
    
    myocardium_data = np.zeros((num_frames, image_size, image_size), dtype=np.float64)
    
    y_coords, x_coords = np.where(myocardium_mask == 1)
    
    time_series = gamma_variate(np.arange(num_frames), A=1000, t0=0, alpha=2, beta=1)
    
    for y, x in zip(y_coords, x_coords):
        myocardium_data[:, y, x] = time_series
        
    return myocardium_data


@pytest.fixture
def temp_dir():
    """
    Create a temporary directory for test files.
    
    Files are written directly to tests/test_outputs/ directory.
    Files will remain after the test for inspection.
    Files are kept after the test for inspection.
    You can manually delete them if needed.
    """

    test_outputs_dir = os.path.join(project_root, 'tests', 'test_outputs')
    os.makedirs(test_outputs_dir, exist_ok=True)
    
    yield test_outputs_dir


@pytest.fixture
def sample_blood_pool_mask():
    """Create a sample 2D binary mask for blood pool region.
    
    Returns a binary mask with a square region set to 1.
    """
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    
    # Create a square region in the center
    center_y, center_x = image_size // 2, image_size // 2
    blood_pool_size = image_size // 4  
    
    y, x = np.ogrid[:image_size, :image_size]
    
    # Create square region: both x and y must be within half_size from center
    mask[(np.abs(x - center_x) <= blood_pool_size/2) & (np.abs(y - center_y) <= blood_pool_size/2)] = 1

    return mask


@pytest.fixture
def sample_myocardium_mask():
    """Create a sample 2D binary mask for myocardium region.
    
    Returns a binary mask padded around the blood pool region.
    """
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    
    # Create a square region in the center
    center_y, center_x = image_size // 2, image_size // 2
    blood_pool_size = image_size // 4
    myocardium_size = image_size // 4  
    
    y, x = np.ogrid[:image_size, :image_size]
    
    # Create larger square for myocardium
    mask[(np.abs(x - center_x) <= myocardium_size/2+blood_pool_size/2) & (np.abs(y - center_y) <= myocardium_size/2+blood_pool_size/2)] = 1

    # Remove blood pool region from myocardium mask
    mask[(np.abs(x - center_x) <= blood_pool_size/2) & (np.abs(y - center_y) <= blood_pool_size/2)] = 0

    return mask


@pytest.fixture
def sample_frames(sample_blood_pool_data, sample_myocardium_data):
    """Create a sample 3D frames array for testing."""
    frames = sample_blood_pool_data + sample_myocardium_data
        
    return frames
