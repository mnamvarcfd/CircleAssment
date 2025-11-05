from scipy.signal import convolve
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from datetime import datetime
from compute_quantity import ComputeQuantity
from logging_config import setup_logger

logger = setup_logger(name='myocardial_blood_flow')

class MyocardialBloodFlow(ComputeQuantity):
    """
    A class for computing the myocardial blood flow
    """
    def __init__(self,
                 frames: np.ndarray,
                 blood_pool_mask: np.ndarray,
                 myo_mask: np.ndarray,
                 tau_d: float = 1.0):
        """
        Args:
            frames (np.ndarray): A 3D NumPy array representing the frames of the DICOM files.
                Its shape is (num_time_stamps, image_size, image_size).
            myo_mask (np.ndarray): A 2D NumPy array representing the myocardial mask.
                Its shape is (image_size, image_size).
            tau_d (float): Delay parameter for the Fermi function, default is 1.0.
        """
        super().__init__(frames=frames, blood_pool_mask=blood_pool_mask, myo_mask=myo_mask)

        self.frames = frames
        self.num_time_stamps = frames.shape[0]
        self.tau_d = tau_d
        self.aif = self.arterial_input_function()
        self.myo_pixel_coordinates, self.myo_pixel_time_series = self.myocardium_time_series()

        self.mbf = np.zeros(self.myo_pixel_time_series.shape[0])
     
    @staticmethod
    def fermi_function(t:np.ndarray,
                       F:float,
                       tau_0:float,
                       k:float,
                       tau_d:float = 1)->np.ndarray:
        """
        Fermi function for impulse response
        The description of the args is based on eq 5 in Jerosch-Herold 1998 paper
        
        Fixed parameter: tau_d (delay). According to Jerosch-Herold 1998 paper, tau_d should be 
        defined by user! So, we hardcode it as 5.
        
        Args:
            t (numpy.ndarray): Time
            F (float): Rate of flow
            tau_0 (float): width of the shoulder of the Fermi function
            k (float): decay rate of Fermi function due to contrast agent washout.
        Returns:
            R_F (numpy.ndarray): Impulse response function (Fermi function)
        """
        
        delayed_t = t - tau_d
        
        step_function = (delayed_t >= 0).astype(np.float64)
        
        exponent = np.exp(k * (delayed_t - tau_0)) + 1.0
        
        R_F = F / exponent * step_function

        return R_F

    @staticmethod
    def calculate_fermi_maximum(F: float, tau_0: float, k: float, tau_d: float) -> float:
        """
        Calculate the maximum value of the Fermi function using fitted parameters.

        The maximum occurs at t = tau_d, where delayed_t = 0.
        At this point: R_F_max = F / (exp(-k * tau_0) + 1.0)

        Args:
            F (float): Flow rate parameter
            tau_0 (float): Shoulder width parameter
            k (float): Decay rate parameter
            tau_d (float): Delay parameter

        Returns:
            float: Maximum value of the Fermi function
        """
        if k == 0:
            # Handle edge case where k = 0
            return F / 2.0

        exponent_at_max = np.exp(-k * tau_0) + 1.0
        R_F_max = F / exponent_at_max

        return R_F_max

    @staticmethod
    def calculate_fermi_maximum_from_popt(popt: np.ndarray, tau_d: float = 1.0) -> float:
        """
        Calculate the maximum value of the Fermi function from fitted parameters array.

        Args:
            popt (np.ndarray): Array of fitted parameters [F, tau_0, k] (tau_d is fixed)
            tau_d (float): Fixed delay parameter, default is 1.0

        Returns:
            float: Maximum value of the Fermi function
        """
        if len(popt) != 3:
            raise ValueError("popt must contain exactly 3 parameters: [F, tau_0, k] (tau_d is fixed)")

        F, tau_0, k = popt
        return MyocardialBloodFlow.calculate_fermi_maximum(F, tau_0, k, tau_d)


    def _fitting_model_function(self,
                                t: np.ndarray,
                                F: float,
                                tau_0: float,
                                k: float) -> np.ndarray:
        """
        This is the model to be fit, representing q(t) = c_in(t) * R_F(t) eq. 3 in Jerosch-Herold 1998
        It returns the convolution of the AIF with the Fermi function.
        So, this function is designed to simulate the time curve for a single pixel.

        Args:
            t (np.ndarray): Time array
            F (float): Rate of flow (parameter to fit)
            tau_0 (float): Shoulder width (parameter to fit)
            k (float): Decay rate (parameter to fit)
            tau_d (float): Delay parameter (taken from instance variable self.tau_d)

        Returns:
            np.ndarray: The modeled myocardial signal (myo_model) for a single pixel (1D array)
        """

        # 1. Generate the impulse response R_F(t)
        R_F = self.fermi_function(t, F, tau_0, k, tau_d=self.tau_d)

        # 2. Convolve with AIF (c_in(t))
        # 'full' mode and slicing ensures the output is the same length as t
        myo_model = convolve(self.aif, R_F, mode='full')[:len(t)]

        return myo_model


    def fermi_curve_fitting(self,
                            myo_time_series: np.ndarray,
                            F_init:float=1.0,
                            tau_0_init:float=20.0,
                            k_init:float=0.1,
                            plot: bool = False,
                            pixel_index: int = 0) -> float:
        """
        Computes the Myocardial Blood Flow (MBF) for a single pixel's time curve.

        This function performs a constrained deconvolution by fitting a convolution
        model (AIF convoluted with a Fermi function) to the measured time-signal
        curve of a single myocardial pixel. The theoretical background is based on 
        Section D of Jerosch-Herold 1998 paper.

        Args:
            MYO_time_series (np.ndarray): A 1D NumPy array representing the signal
                intensity over time for a single myocardial pixel (q(t)).
            F_init (float): Initial guess for flow rate (F), default is 1.0
            tau_0_init (float): Initial guess for shoulder width (tau_0), default is 20.0
            k_init (float): Initial guess for decay rate (k), default is 0.1
            plot (bool): Whether to generate and save a plot of the fitted function, default is False
            pixel_index (int): Index of the pixel being processed, used for filename generation, default is 0
        Returns:
            float: The calculated Myocardial Blood Flow (MBF) for the pixel,
                which corresponds to the fitted 'F' parameter for the Fermi function.
                Returns 0.0 if the curve fitting fails to converge.
        """
        
        # Create the time array (independent variable)
        t = np.arange(self.num_time_stamps)

        try:
            # Fit the convolution_model to the measured MYO_pixel data
            # the documentation of curve_fit is based on the following link:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
            popt, pcov = curve_fit(
                self._fitting_model_function,
                t,                # xdata (independent variable)
                myo_time_series,        # ydata (measured 1D pixel curve)
                p0=[F_init, tau_0_init, k_init], # initial parameter guesses
                method='lm',  # 'lm' isLevenberg-Marquardt algorithm choosed baed on Jerosch-Herold 1998 paper
                maxfev=1000000
            )
            
            logger.debug(f"popt: {popt}")

            #This is only for presentation =================================================
            if plot:
                from save_data_manager import SaveDataManager
                manager = SaveDataManager(results_dir='F:/18_Circle/code/tests/results/workflow_real')
                series = self.fermi_function(t, popt[0], popt[1], popt[2], tau_d=self.tau_d)

                manager.plot_pixel_over_time(
                    series,
                    title="Fermi Function Fitting",
                    y_label="Value",
                    output_filename=f"fermi_function_fitting_pixel_{pixel_index}.png"
                )
            #==============================================================================
        
            # popt contains the best-fit parameters [F, tau_0, k], tau_d is stored in self.tau_d
            MBF = self.calculate_fermi_maximum_from_popt(popt, tau_d=self.tau_d)
        
        except RuntimeError:
            # If curve_fit fails to converge, return 0.0 for this pixel.
            MBF = -1.0
        
        return MBF


    def compute(self) -> np.ndarray:

        # Iterate over each myocardial pixel's time series
        for i, myo_time_series in enumerate(self.myo_pixel_time_series):
            # Calculate MBF for this single pixel and generate plot for all pixels
            mbf_value = self.fermi_curve_fitting(myo_time_series, F_init=1.0, tau_0_init=20.0, k_init=0.1, plot=False, pixel_index=i)

            logger.debug(f"mbf_value {i}: {mbf_value}")
            self.mbf[i] = mbf_value

        return self.mbf


  