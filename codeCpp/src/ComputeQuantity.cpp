#include "ComputeQuantity.h"
#include <iostream>
#include <numeric>

ComputeQuantity::ComputeQuantity(const std::vector<cv::Mat>& frames,
                                 const cv::Mat& bloodPoolMask,
                                 const cv::Mat& myoMask)
    : frames_(frames), bloodPoolMask_(bloodPoolMask), myoMask_(myoMask) {}

std::tuple<std::vector<cv::Point>, cv::Mat>
ComputeQuantity::pixelTimeSeries(const cv::Mat& mask) {
    std::vector<cv::Point> coordinates;
    for (int y = 0; y < mask.rows; ++y)
        for (int x = 0; x < mask.cols; ++x)
            if (mask.at<uchar>(y, x) == 1)
                coordinates.emplace_back(x, y);

    cv::Mat timeSeries((int)coordinates.size(), (int)frames_.size(), CV_64F);

    for (size_t i = 0; i < coordinates.size(); ++i) {
        for (size_t t = 0; t < frames_.size(); ++t) {
            timeSeries.at<double>(i, t) =
                frames_[t].at<uchar>(coordinates[i].y, coordinates[i].x);
        }
    }
    return {coordinates, timeSeries};
}

std::tuple<std::vector<cv::Point>, cv::Mat>
ComputeQuantity::myocardiumTimeSeries() {
    return pixelTimeSeries(myoMask_);
}

std::tuple<std::vector<cv::Point>, cv::Mat>
ComputeQuantity::bloodPoolTimeSeries() {
    return pixelTimeSeries(bloodPoolMask_);
}

cv::Mat ComputeQuantity::arterialInputFunction() {
    auto [coords, timeSeries] = bloodPoolTimeSeries();
    cv::Mat aif;
    cv::reduce(timeSeries, aif, 0, cv::REDUCE_AVG);
    return aif;
}
