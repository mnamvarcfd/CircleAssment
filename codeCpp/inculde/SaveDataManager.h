#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

class SaveDataManager {
public:
    SaveDataManager(const std::string& resultsDir = "results");

    // Save pixel values as a 2D image map using a mask for spatial reference
    std::string saveImage(const std::vector<double>& pixelValues,
                          const cv::Mat& mask,
                          const std::string& valueTitle,
                          const std::string& outputFilename);

    // Create a movie from frames array (t, h, w)
    std::string createMovieFromFrames(const std::vector<cv::Mat>& frames,
                                      const std::string& outputPath = "dicom_movie.avi",
                                      int fps = 10,
                                      bool normalize = true,
                                      bool applyColormap = true,
                                      const std::string& codec = "MJPG");

    // Plot a pixel's value over time and save as image
    std::string plotPixelOverTime(const std::vector<double>& series,
                                   const std::string& title = "Pixel Timeseries",
                                   const std::string& yLabel = "Signal Intensity",
                                   const std::string& outputFilename = "pixel_timeseries.png");

    // Plot and save a 2D mask as a PNG file
    std::string plotMask(const cv::Mat& mask,
                         const std::string& outputFilename = "mask.png");

private:
    std::string resultsDir;

    // Helper: Create directory if it doesn't exist
    void ensureDirectoryExists(const std::string& path);

    // Helper: Draw text on image
    void putText(cv::Mat& img, const std::string& text, cv::Point pos, 
                 double fontScale = 0.5, cv::Scalar color = cv::Scalar(255, 255, 255));
};

