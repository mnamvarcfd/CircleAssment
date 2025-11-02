from scipy.signal import convolve
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from compute_quantity import ComputeQuantity
from logging_config import setup_logger

logger = setup_logger(name='myocardial_blood_flow')

class MyocardialBloodFlow:
    """
    A class for computing the myocardial blood flow
    """
    def __init__(self, aif: np.ndarray, myo: np.ndarray, myo_mask: np.ndarray):
        """
        Args:
            
            AIF (np.ndarray): A 1D NumPy array representing the Arterial Input
                            Function (c_in(t)). Its length must match the time
                            dimension of the MYO_pixel.
            myo (np.ndarray): A 3D NumPy array representing the myocardial time series at each pixel
                              Its shape is (num_time_stamps, image_size, image_size).
        """
        self.aif = aif
        self.myo = myo
        self.myo_mask = myo_mask
        
        self.num_time_stamps =  self.myo.shape[0]
        
        self.num_pixels_in_x_axis =  self.myo.shape[1]
        self.num_pixels_in_y_axis =  self.myo.shape[2]

        self.mbf= np.zeros((self.num_pixels_in_x_axis, self.num_pixels_in_y_axis))
     
        
        logger.debug(f"AIF(t): {self.aif}")
 
        for x in range(self.num_pixels_in_x_axis):
            for y in range(self.num_pixels_in_y_axis):
                if self.myo_mask[x, y] == 0:
                    continue
                myo_pixel_curve = self.myo[:, x, y]
                logger.debug(f"myo_pixel_curve: {x}, {y}: {myo_pixel_curve}")
        
    def _fermi_function(self, 
                        t:np.ndarray, 
                        F:float, 
                        tau_0:float, 
                        k:float)->np.ndarray:
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
        tau_d = 1
        
        delayed_t = t - tau_d
        
        step_function = (delayed_t >= 0).astype(np.float64)
        
        exponent = np.exp(k * (delayed_t - tau_0)) + 1.0
        
        R_F = F / exponent * step_function
        
        return R_F


    def _convolution_model(self,
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

        Returns:
            np.ndarray: The modeled myocardial signal (myo_model) for a single pixel (1D array)
        """

        # 1. Generate the impulse response R_F(t)
        R_F = self._fermi_function(t, F, tau_0, k)

        # 2. Convolve with AIF (c_in(t))
        # 'full' mode and slicing ensures the output is the same length as t
        myo_model = convolve(self.aif, R_F, mode='full')[:len(t)]

        return myo_model


    def fermi_function_fitting(self, 
                               MYO_time_series: np.ndarray, 
                               F_init:float=1.0, 
                               tau_0_init:float=20.0, 
                               k_init:float=0.1) -> float:
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
        Returns:
            float: The calculated Myocardial Blood Flow (MBF) for the pixel,
                which corresponds to the fitted 'F' parameter for the Fermi function.
                Returns 0.0 if the curve fitting fails to converge.
        """
        
        # Create the time array (independent variable)
        t = np.arange(self.num_time_stamps)

        # Create a lambda function for the model to be passed to curve_fit.
        # 'time' is the independent variable and the parameters to be optimized (F, tau_0, k).
        model_to_fit = lambda time, F, tau_0, k: self._convolution_model(time, F, tau_0, k)
        
        try:
            # Fit the convolution_model to the measured MYO_pixel data
            # the documentation of curve_fit is based on the following link:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
            popt, pcov = curve_fit(
                model_to_fit, 
                t,                # xdata (independent variable)
                MYO_time_series,        # ydata (measured 1D pixel curve)
                p0=[F_init, tau_0_init, k_init], # initial parameter guesses
                method='lm',  # 'lm' isLevenberg-Marquardt algorithm choosed baed on Jerosch-Herold 1998 paper
                maxfev=1000
            )
            
            logger.debug(f"popt: {popt}")
            
            #This is only for presentation =================================================
            from save_data_manager import SaveDataManager
            manager = SaveDataManager(results_dir='F:/18_Circle/code/tests/test_outputs')
            series = self._fermi_function(t, popt[0], popt[1], popt[2])
            
            manager.plot_pixel_over_time(
                series,
                title="Fermi Function Fitting",
                y_label="Value",
                output_filename="fermi_function_fitting.png"
            )
            #==============================================================================
        
            # popt contains the best-fit parameters [F, tau_0, k]
            MBF = popt[0]
        
        except RuntimeError:
            # If curve_fit fails to converge, return 0.0 for this pixel.
            MBF = 0.0
        
        return MBF


    def compute(self) -> np.ndarray:
            
        for x in range(self.num_pixels_in_x_axis):
            for y in range(self.num_pixels_in_y_axis):
                if self.myo_mask[x, y] == 0:
                    continue
                myo_pixel_curve = self.myo[:, x, y]

                # Calculate MBF for this single pixel
                mbf_value = self.fermi_function_fitting(myo_pixel_curve, F_init=3.0, tau_0_init=1.0, k_init=0.1)
                
                logger.debug(f"mbf_value: {x}, {y}: {mbf_value}") 
                self.mbf[x, y] = mbf_value
        
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

    # some statistics
    logger.info(f"MBF min: {mbf.min():.3f}")
    logger.info(f"MBF max: {mbf.max():.3f}")
    logger.info(f"MBF mean: {mbf.mean():.3f}")
    