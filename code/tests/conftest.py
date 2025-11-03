"""
Shared pytest fixtures for testing.
"""
import os
import sys

# Add parent directory to path to allow importing src modules
# This must happen before any other imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import tempfile
import shutil
from scipy.signal import convolve
from tests.test_utils import gamma_variate


num_frames = 60
image_size = 50

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
def sample_mask():
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
def sample_frames():
    """Create a sample 3D frames array for testing."""
    frames = np.zeros((num_frames, image_size, image_size), dtype=np.float64)
    
    for t in range(num_frames):
        for x in range(image_size):
            for y in range(image_size):
                frames[t, y, x] = t+x+y
    
    return frames


@pytest.fixture
def sample_myocardium_mask():
    """Create a sample 2D binary mask for myocardium region."""
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    
    mask[2:3, 2:3] = 1  # Set a small region in the center to 1

    return mask


@pytest.fixture
def sample_blood_pool_mask():
    """Create a sample 2D binary mask for blood pool region."""
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    
    mask[1:2, 1:2] = 1  # Set a small region in the center to 1
    
    return mask


@pytest.fixture
def sample_aif(sample_frames, sample_blood_pool_mask):
    """Create a sample AIF time series."""
    aif = np.zeros((num_frames,), dtype=np.float64)
    
    for t in range(num_frames):
        for x in range(image_size):
            for y in range(image_size):
                if sample_blood_pool_mask[y, x] == 1:
                    aif[t] += sample_frames[t, y, x]
    
    aif = aif / np.sum(sample_blood_pool_mask == 1)
    
    return aif
