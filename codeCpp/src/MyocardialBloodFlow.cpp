#include "MyocardialBloodFlow.h"
#include "logger.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>
#include <limits>
#include "optimization.h"

MyocardialBloodFlow::MyocardialBloodFlow(const std::vector<cv::Mat>& frames,
                                         const cv::Mat& bloodPoolMask,
                                         const cv::Mat& myoMask)
    : ComputeQuantity(frames, bloodPoolMask, myoMask)
    , frames_(frames)
    , numTimeStamps_(static_cast<int>(frames.size()))
{
    // Get AIF from base class
    aif_ = arterialInputFunction();

    // Get myocardium time series from base class
    auto [coords, timeSeries] = myocardiumTimeSeries();
    myoPixelCoordinates_ = coords;
    myoPixelTimeSeries_ = timeSeries;

    // Initialize MBF array
    mbf_ = cv::Mat::zeros(myoPixelTimeSeries_.rows, 1, CV_64F);
}

cv::Mat MyocardialBloodFlow::fermiFunction(const cv::Mat& t,
                                          double F,
                                          double tau_0,
                                          double k,
                                          double tau_d)
{
    cv::Mat delayed_t;
    t.copyTo(delayed_t);
    delayed_t -= tau_d;

    // Step function: (delayed_t >= 0) ? 1.0 : 0.0
    cv::Mat stepFunction = (delayed_t >= 0) / 255.0;
    stepFunction.convertTo(stepFunction, CV_64F);

    // Compute exponent: exp(k * (delayed_t - tau_0)) + 1.0
    cv::Mat exponent;
    cv::Mat temp = delayed_t - tau_0;
    temp *= k;
    cv::exp(temp, exponent);
    exponent += 1.0;

    // R_F = F / exponent * stepFunction
    cv::Mat R_F;
    cv::divide(F, exponent, R_F);
    R_F = R_F.mul(stepFunction);

    return R_F;
}

cv::Mat MyocardialBloodFlow::convolve(const cv::Mat& a, const cv::Mat& b) const
{
    // Ensure both are 1D column vectors
    cv::Mat a_1d = a.reshape(1, a.total());
    cv::Mat b_1d = b.reshape(1, b.total());

    int aLen = a_1d.rows;
    int bLen = b_1d.rows;
    int resultLen = aLen + bLen - 1;

    cv::Mat result = cv::Mat::zeros(resultLen, 1, CV_64F);

    for (int i = 0; i < resultLen; ++i) {
        double sum = 0.0;
        for (int j = 0; j < bLen; ++j) {
            int idx = i - j;
            if (idx >= 0 && idx < aLen) {
                sum += a_1d.at<double>(idx, 0) * b_1d.at<double>(j, 0);
            }
        }
        result.at<double>(i, 0) = sum;
    }

    return result;
}

cv::Mat MyocardialBloodFlow::fittingModelFunction(const cv::Mat& t,
                                                  double F,
                                                  double tau_0,
                                                  double k) const
{
    // Generate the impulse response R_F(t)
    cv::Mat R_F = fermiFunction(t, F, tau_0, k);

    // Convolve with AIF (c_in(t))
    cv::Mat fullConvolution = convolve(aif_, R_F);

    // Extract the first len(t) elements
    cv::Mat myoModel = fullConvolution(cv::Rect(0, 0, 1, t.rows)).clone();

    return myoModel;
}

// Global variables for ALGLIB optimization
static std::vector<double> global_t;
static std::vector<double> global_observed;
static std::vector<double> global_aif;

// Objective function for ALGLIB optimization
void fermi_fitting_function(const alglib::real_1d_array &x, double &func, void *ptr) {
    // x[0] = F, x[1] = tau_0, x[2] = k
    double F = x[0];
    double tau_0 = x[1];
    double k = x[2];
    double tau_d = 1.0;

    // Compute Fermi function for each time point
    std::vector<double> R_F(global_t.size());
    for (size_t i = 0; i < global_t.size(); ++i) {
        double delayed_t = global_t[i] - tau_d;
        double step = (delayed_t >= 0.0) ? 1.0 : 0.0;
        double exponent = std::exp(k * (delayed_t - tau_0)) + 1.0;
        R_F[i] = F / exponent * step;
    }

    // Convolve AIF with R_F
    std::vector<double> model(global_t.size());
    for (size_t i = 0; i < global_t.size(); ++i) {
        double sum = 0.0;
        for (size_t j = 0; j <= i && j < R_F.size(); ++j) {
            size_t aif_idx = i - j;
            if (aif_idx < global_aif.size()) {
                sum += global_aif[aif_idx] * R_F[j];
            }
        }
        model[i] = sum;
    }

    // Compute sum of squared residuals
    func = 0.0;
    for (size_t i = 0; i < global_t.size(); ++i) {
        double residual = model[i] - global_observed[i];
        func += residual * residual;
    }
}

double MyocardialBloodFlow::fermiCurveFitting(const cv::Mat& myoTimeSeries,
                                              double F_init,
                                              double tau_0_init,
                                              double k_init) const
{
    // Create time array as vector
    std::vector<double> t(numTimeStamps_);
    for (int i = 0; i < numTimeStamps_; ++i) {
        t[i] = static_cast<double>(i);
    }

    // Convert myoTimeSeries to vector
    cv::Mat myoSeries = myoTimeSeries.reshape(1, myoTimeSeries.total());
    std::vector<double> observed(numTimeStamps_);
    for (int i = 0; i < numTimeStamps_; ++i) {
        observed[i] = myoSeries.at<double>(i, 0);
    }

    // Convert AIF to vector (AIF is a row vector from reduce operation)
    std::vector<double> aif(numTimeStamps_);
    if (aif_.rows == 1) {
        // Row vector
        for (int i = 0; i < numTimeStamps_ && i < aif_.cols; ++i) {
            aif[i] = aif_.at<double>(0, i);
        }
    } else {
        // Column vector
        for (int i = 0; i < numTimeStamps_ && i < aif_.rows; ++i) {
            aif[i] = aif_.at<double>(i, 0);
        }
    }

    try {
        // Set global variables for ALGLIB
        global_t = t;
        global_observed = observed;
        global_aif = aif;

        // Set up ALGLIB optimization
        alglib::real_1d_array x;
        x.setlength(3);
        x[0] = F_init;
        x[1] = tau_0_init;
        x[2] = k_init;
        
        // Set bounds: F >= 0, tau_0 >= 0, k >= 0
        alglib::real_1d_array bndl;
        bndl.setlength(3);
        bndl[0] = 0.0;
        bndl[1] = 0.0;
        bndl[2] = 0.0;
        
        alglib::real_1d_array bndu;
        bndu.setlength(3);
        bndu[0] = 1e10;  // Large positive number instead of +inf
        bndu[1] = 1e10;
        bndu[2] = 1e10;

        // Create optimizer
        alglib::minbleicstate state;
        alglib::minbleiccreate(x, state);
        alglib::minbleicsetbc(state, bndl, bndu);
        alglib::minbleicsetcond(state, 0.0, 0.0, 0.0, 1000);

        // Optimize
        alglib::minbleicoptimize(state, fermi_fitting_function);
        
        // Get results
        alglib::minbleicreport rep;
        alglib::minbleicresults(state, x, rep);

        if (rep.terminationtype <= 0) {
            Logger::get()->debug("Curve fitting did not converge. Termination type: " + 
                                std::to_string(rep.terminationtype));
            return 0.0;
        }

        Logger::get()->debug("Fitted parameters: F=" + std::to_string(x[0]) +
                             ", tau_0=" + std::to_string(x[1]) +
                             ", k=" + std::to_string(x[2]) +
                             ", iterations=" + std::to_string(rep.iterationscount));

        // MBF is the fitted F parameter
        return x[0];

    } catch (const std::exception& e) {
        Logger::get()->error("Curve fitting exception: " + std::string(e.what()));
        return 0.0;
    }
}

cv::Mat MyocardialBloodFlow::compute()
{
    // Iterate over each myocardial pixel's time series
    for (int i = 0; i < myoPixelTimeSeries_.rows; ++i) {
        // Extract time series for this pixel
        cv::Mat pixelSeries = myoPixelTimeSeries_.row(i).t();

        // Calculate MBF for this single pixel
        double mbfValue = fermiCurveFitting(pixelSeries, 1.0, 20.0, 0.1);

        Logger::get()->debug("mbf_value " + std::to_string(i) + ": " + std::to_string(mbfValue));

        mbf_.at<double>(i, 0) = mbfValue;
    }

    return mbf_;
}

