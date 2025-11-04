#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>

class ComputeQuantity {
public:
    ComputeQuantity(const std::vector<cv::Mat>& frames,
                    const cv::Mat& bloodPoolMask,
                    const cv::Mat& myoMask);

    std::tuple<std::vector<cv::Point>, cv::Mat> pixelTimeSeries(const cv::Mat& mask);
    std::tuple<std::vector<cv::Point>, cv::Mat> myocardiumTimeSeries();
    std::tuple<std::vector<cv::Point>, cv::Mat> bloodPoolTimeSeries();
    cv::Mat arterialInputFunction();

private:
    std::vector<cv::Mat> frames_;
    cv::Mat bloodPoolMask_;
    cv::Mat myoMask_;
};
