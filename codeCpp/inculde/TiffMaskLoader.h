#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>

class TiffMaskLoader {
public:
    TiffMaskLoader(const std::string& filename);

    // Load mask at the specified index (0 for first page, 1 for second, etc.)
    cv::Mat loadMask(int maskIndex = 0);

private:
    std::string maskFile;
    std::vector<cv::Mat> tiffPages;
    void loadAllPages();
};
