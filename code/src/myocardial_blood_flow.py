from scipy.signal import convolve
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from compute_quantity import ComputeQuantity
from logging_config import setup_logger

logger = setup_logger()

class MyocardialBloodFlow:
    """
    A class for computing the myocardial blood flow
    """
    def __init__(self, aif: np.ndarray, myo: np.ndarray):
        """
        Args:
            
            AIF (np.ndarray): A 1D NumPy array representing the Arterial Input
                            Function (c_in(t)). Its length must match the time
                            dimension of the MYO_pixel.
            myo (np.ndarray): A 2D NumPy array representing the myocardial time series
        """
        self.aif = aif
        self.myo = myo

        self.mbf = None  # This holds the MBF value for each pixel
        
        
    def _fermi_function(self, 
                        t:np.ndarray, 
                        F:float, 
                        tau_0:float, 
                        k:float, 
                        tau_d:float)->np.ndarray:
        """
        Fermi function for impulse response
        The description of the args is based on eq 5 in Jerosch-Herold 1998 paper
        
        Args:
            t (numpy.ndarray): Time
            F (float): Rate of flow
            tau_0 (float): width of the shoulder of the Fermi function
            k (float): decay rate of Fermi function due to contrast agent washout.
            tau_d (float): delay
        Returns:
            R_F (numpy.ndarray): Impulse response function (Fermi function)
        """
        
        delayed_t = t - tau_d
        
        step_function = (delayed_t >= 0).astype(np.float64)
        
        exponent = np.exp(k * (delayed_t - tau_0)) + 1.0
        
        R_F = F / exponent * step_function
        
        return R_F


    def _convolution_model(self,
                           t: np.ndarray,
                            F: float,
                            tau_0: float,
                            k: float,
                            tau_d: float) -> np.ndarray:
        """
        This is the model to be fit, representing q(t) = c_in(t) * R_F(t) eq. 3 in Jerosch-Herold 1998
        It returns the convolution of the AIF with the Fermi function.
        So, this function is designed to simulate the time curve for a single pixel.

        Args:
            t (np.ndarray): Time array
            F (float): Rate of flow (parameter to fit)
            tau_0 (float): Shoulder width (parameter to fit)
            k (float): Decay rate (parameter to fit)
            tau_d (float): Time delay (fixed value)

        Returns:
            np.ndarray: The modeled myocardial signal (myo_model) for a single pixel (1D array)
        """

        # 1. Generate the impulse response R_F(t)
        R_F = self._fermi_function(t, F, tau_0, k, tau_d)

        # 2. Convolve with AIF (c_in(t))
        # 'full' mode and slicing ensures the output is the same length as t
        myo_model = convolve(self.aif, R_F, mode='full')[:len(t)]

        return myo_model


    def _MBF(self, MYO_pixel: np.ndarray) -> float:
        """
        Computes the Myocardial Blood Flow (MBF) for a single pixel's time curve.

        This function performs a constrained deconvolution by fitting a convolution
        model (AIF convoluted with a Fermi function) to the measured time-signal
        curve of a single myocardial pixel. The theoretical background is based on 
        Section D of Jerosch-Herold 1998 paper.

        Args:
            MYO_pixel (np.ndarray): A 1D NumPy array representing the signal
                                    intensity over time for a single
                                    myocardial pixel (q(t)).

        Returns:
            float: The calculated Myocardial Blood Flow (MBF) for the pixel,
                which corresponds to the fitted 'F' parameter for the Fermi function.
                Returns 0.0 if the curve fitting fails to converge.
        """
        
        # Initial guesses for parameters to be fitted
        F_init = 1.0      # Initial guess for flow (F)
        tau_0_init = 1.0  # Initial guess for shoulder width (tau_0)
        k_init = 0.5      # Initial guess for decay rate (k)
        
        # Fixed parameter: tau_d (delay). According to Jerosch-Herold 1998 paper, tau_d should be 
        # defined by user! So, we hardcode it as 0.01.
        tau_d = 0.01 
        
        # Create the time array (independent variable)
        t = np.arange(len(self.aif))

        # Create a lambda function for the model to be passed to curve_fit.
        # This "fixes" the 'tau_d' argument, leaving only
        # 't' as the independent variable and the parameters to be optimized (F, tau_0, k).
        model_to_fit = lambda t_data, F, tau_0, k: self._convolution_model(t_data, F, tau_0, k, tau_d)
        
        
        # To prevent unrealistic parameters, we add bounds to the parameters.
        # All the parameters are based on the paper simulations from Jerosch-Herold 1998, pages 5-6:
        bounds = (
                [
                   0.5,   # F_min: Minimum flow from paper simulations (page 5: "flows from 0.5 to 4.0 ml/min/g")
                   0.1,   # tau_0_min: Smallest reasonable width
                   0.1,   # k_min: Smallest reasonable decay rate
                ],
                [
                   4.0,   # F_max: Maximum flow from paper simulations (page 5)
                   10.0,  # tau_0_max: Largest reasonable width  
                   2.0    # k_max: Largest reasonable decay rate (paper used up to 5.0 in simulations but 2.0 is more physiological)
                ]
                )
        
        try:
            # Fit the convolution_model to the measured MYO_pixel data
            # the documentation of curve_fit is based on the following link:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
            popt, pcov = curve_fit(
                model_to_fit, 
                t,                # xdata (independent variable)
                MYO_pixel,        # ydata (measured 1D pixel curve)
                p0=[F_init, tau_0_init, k_init], # initial parameter guesses
                bounds=bounds,
                maxfev=10000
            )

            # popt contains the best-fit parameters [F, tau_0, k]
            # The MBF is the fitted flow parameter F. 
            MBF = popt[0]
        
        except RuntimeError:
            # If curve_fit fails to converge, return 0.0 for this pixel.
            MBF = 0.0
        
        return MBF


    def compute(self) -> np.ndarray:
        
        num_timepoints, num_pixels = self.myo.shape
        
        mbf_values = []
        
        for i in range(num_pixels):
            myo_pixel_curve = self.myo[:, i]

            # Calculate MBF for this single pixel
            mbf_value = self._MBF(myo_pixel_curve)
            
            mbf_values.append(mbf_value)
        
        # Convert the final list to a numpy array
        self.mbf = np.array(mbf_values)
        
        return self.mbf


    
if __name__ == "__main__":
    
    from data_loader import DataLoader
    data_loader = DataLoader(dicom_dir="input_data/DICOM_files")
    frames = data_loader.dicom()
    aif_mask = data_loader.mask(mask_index=0)  # Blood pool
    myo_mask = data_loader.mask(mask_index=1)  # Myocardium
    
    compute_quantity = ComputeQuantity(frames=frames, aif_mask=aif_mask, myo_mask=myo_mask)
    aif = compute_quantity.arterial_input_function()
    myo_pixel_coordinates, myo_time_series = compute_quantity.myocardium_time_series()
    
    myocardial_blood_flow = MyocardialBloodFlow(aif=aif, myo=myo_time_series.T)
    mbf = myocardial_blood_flow.compute()

    from save_data_manager import SaveDataManager
    save_data_manager = SaveDataManager()
    save_data_manager.save_image(mbf, myo_mask, Value_title="MBF", output_filename="mbf_map.png")
    
    # Save MBF results to CSV
    mbf_df = pd.DataFrame(mbf, columns=['MBF'])
    mbf_df.to_csv('results/mbf_results.csv', index=False)
    logger.info(f"MBF computed for {len(mbf)} pixels and saved to results/mbf_results.csv")

    # Print some statistics
    logger.info(f"MBF min: {mbf.min():.3f}")
    logger.info(f"MBF max: {mbf.max():.3f}")
    logger.info(f"MBF mean: {mbf.mean():.3f}")
    