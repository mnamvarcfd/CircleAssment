#pragma once

#include "ComputeQuantity.h"
#include <opencv2/opencv.hpp>
#include <vector>

class MyocardialBloodFlow : public ComputeQuantity {
public:
    MyocardialBloodFlow(const std::vector<cv::Mat>& frames,
                       const cv::Mat& bloodPoolMask,
                       const cv::Mat& myoMask);

    // Static Fermi function for impulse response
    static cv::Mat fermiFunction(const cv::Mat& t,
                                 double F,
                                 double tau_0,
                                 double k,
                                 double tau_d = 1.0);

    // Fitting model function: convolution of AIF with Fermi function
    cv::Mat fittingModelFunction(const cv::Mat& t,
                                 double F,
                                 double tau_0,
                                 double k) const;

    // Perform Fermi curve fitting for a single pixel's time series
    double fermiCurveFitting(const cv::Mat& myoTimeSeries,
                            double F_init = 1.0,
                            double tau_0_init = 20.0,
                            double k_init = 0.1) const;

    // Compute MBF for all pixels
    cv::Mat compute();

    // Get computed MBF values
    const cv::Mat& getMBF() const { return mbf_; }

private:
    std::vector<cv::Mat> frames_;
    int numTimeStamps_;
    cv::Mat aif_;
    std::vector<cv::Point> myoPixelCoordinates_;
    cv::Mat myoPixelTimeSeries_;
    cv::Mat mbf_;

    // Helper: Convolution function
    cv::Mat convolve(const cv::Mat& a, const cv::Mat& b) const;
};

