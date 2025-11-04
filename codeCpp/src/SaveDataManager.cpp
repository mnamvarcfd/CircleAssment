#include "SaveDataManager.h"
#include "logger.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>

SaveDataManager::SaveDataManager(const std::string& resultsDir)
    : resultsDir(resultsDir)
{
    ensureDirectoryExists(resultsDir);
    Logger::get()->info("SaveDataManager initialized with results directory: " + resultsDir);
}

void SaveDataManager::ensureDirectoryExists(const std::string& path)
{
    if (!fs::exists(path)) {
        fs::create_directories(path);
        Logger::get()->info("Created results directory: " + path);
    } else {
        Logger::get()->debug("Results directory already exists: " + path);
    }
}

void SaveDataManager::putText(cv::Mat& img, const std::string& text, cv::Point pos,
                              double fontScale, cv::Scalar color)
{
    cv::putText(img, text, pos, cv::FONT_HERSHEY_SIMPLEX, fontScale, color, 1, cv::LINE_AA);
}

std::string SaveDataManager::saveImage(const std::vector<double>& pixelValues,
                                       const cv::Mat& mask,
                                       const std::string& valueTitle,
                                       const std::string& outputFilename)
{
    if (pixelValues.empty() || mask.empty()) {
        Logger::get()->error("saveImage: pixelValues or mask is empty");
        return "";
    }

    if (mask.type() != CV_8UC1) {
        Logger::get()->error("saveImage: mask must be CV_8UC1 (binary mask)");
        return "";
    }

    cv::Size maskShape = mask.size();
    int height = maskShape.height;
    int width = maskShape.width;

    // Get pixel coordinates where mask == 1
    std::vector<cv::Point> coords;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (mask.at<uchar>(y, x) == 1) {
                coords.push_back(cv::Point(x, y));
            }
        }
    }

    // Validate that the number of pixels matches
    if (pixelValues.size() != coords.size()) {
        Logger::get()->error("saveImage: pixelValues has " + std::to_string(pixelValues.size()) +
                             " pixels, but mask has " + std::to_string(coords.size()) + " pixels");
        return "";
    }

    // Create a 2D array for visualization (filled with NaN equivalent: -1 for invalid pixels)
    cv::Mat image2d = cv::Mat::zeros(height, width, CV_64F);
    image2d.setTo(-1); // Use -1 to indicate invalid pixels (outside mask)

    // Fill in the pixel values at their corresponding pixel locations
    for (size_t i = 0; i < coords.size(); ++i) {
        image2d.at<double>(coords[i].y, coords[i].x) = pixelValues[i];
    }

    // Find valid range (exclude -1 and zeros from failed fits)
    double minVal = std::numeric_limits<double>::max();
    double maxVal = std::numeric_limits<double>::lowest();
    double sum = 0.0;
    int validCount = 0;

    for (size_t i = 0; i < pixelValues.size(); ++i) {
        if (pixelValues[i] > 0) {
            minVal = std::min(minVal, pixelValues[i]);
            maxVal = std::max(maxVal, pixelValues[i]);
            sum += pixelValues[i];
            validCount++;
        }
    }

    // Normalize to 0-255 range for visualization
    cv::Mat normalized;
    if (maxVal > minVal) {
        image2d.copyTo(normalized);
        normalized.setTo(0, normalized < 0); // Set invalid pixels to 0
        normalized = (normalized - minVal) / (maxVal - minVal) * 255.0;
        normalized.convertTo(normalized, CV_8UC1);
    } else {
        normalized = cv::Mat::zeros(height, width, CV_8UC1);
    }

    // Apply colormap (hot colormap equivalent)
    cv::Mat colored;
    cv::applyColorMap(normalized, colored, cv::COLORMAP_HOT);

    // Draw statistics text
    std::stringstream stats;
    if (validCount > 0) {
        double mean = sum / validCount;
        stats << "Mean: " << std::fixed << std::setprecision(2) << mean << "\n";
        stats << "Max: " << std::fixed << std::setprecision(2) << maxVal << "\n";
        stats << "Min: " << std::fixed << std::setprecision(2) << minVal;
        
        // Draw text with background
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(stats, line)) {
            lines.push_back(line);
        }
        
        int yPos = 30;
        for (const auto& text : lines) {
            cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
            cv::rectangle(colored, cv::Point(10, yPos - 20), 
                         cv::Point(10 + textSize.width + 10, yPos + 5),
                         cv::Scalar(255, 255, 255), -1);
            putText(colored, text, cv::Point(15, yPos), 0.6, cv::Scalar(0, 0, 0));
            yPos += 25;
        }
    }

    // Add title
    std::string title = "Distribution of " + valueTitle;
    cv::Size titleSize = cv::getTextSize(title, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, nullptr);
    int titleX = (width - titleSize.width) / 2;
    putText(colored, title, cv::Point(titleX, 30), 0.8, cv::Scalar(255, 255, 255));

    // Save the image
    std::string outputFile = (fs::path(resultsDir) / outputFilename).string();
    try {
        cv::imwrite(outputFile, colored);
        Logger::get()->info("Image map saved to: " + outputFile);
        return outputFile;
    } catch (const cv::Exception& e) {
        Logger::get()->error("saveImage: error saving image: " + std::string(e.what()));
        return "";
    }
}

std::string SaveDataManager::createMovieFromFrames(const std::vector<cv::Mat>& frames,
                                                    const std::string& outputPath,
                                                    int fps,
                                                    bool normalize,
                                                    bool applyColormap,
                                                    const std::string& codec)
{
    if (frames.empty()) {
        Logger::get()->error("createMovieFromFrames: frames vector is empty");
        return "";
    }

    int tLen = frames.size();
    int height = frames[0].rows;
    int width = frames[0].cols;

    std::string fullOutputPath = (fs::path(resultsDir) / outputPath).string();

    // Create VideoWriter
    int fourcc = cv::VideoWriter::fourcc(codec[0], codec[1], codec[2], codec[3]);
    cv::VideoWriter video(fullOutputPath, fourcc, fps, cv::Size(width, height), true);

    if (!video.isOpened()) {
        Logger::get()->error("Failed to open video writer");
        return "";
    }

    // Determine normalization range
    double minVal = 0.0, maxVal = 255.0;
    if (normalize) {
        minVal = std::numeric_limits<double>::max();
        maxVal = std::numeric_limits<double>::lowest();
        for (const auto& frame : frames) {
            double fMin, fMax;
            cv::minMaxLoc(frame, &fMin, &fMax);
            minVal = std::min(minVal, fMin);
            maxVal = std::max(maxVal, fMax);
        }
    }

    // Create the movie by traversing through the frames
    for (int t = 0; t < tLen; ++t) {
        cv::Mat arr = frames[t].clone();

        cv::Mat norm;
        if (normalize) {
            // Handle case where all pixels have the same value
            if (maxVal == minVal) {
                norm = cv::Mat::ones(height, width, CV_8UC1) * 127; // Middle gray
            } else {
                arr.convertTo(norm, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
            }
        } else {
            arr.convertTo(norm, CV_8UC1);
        }

        cv::Mat frame;
        if (applyColormap) {
            cv::applyColorMap(norm, frame, cv::COLORMAP_TURBO);
        } else {
            cv::cvtColor(norm, frame, cv::COLOR_GRAY2BGR);
        }

        video.write(frame);
    }

    video.release();
    Logger::get()->info("Movie saved: " + fullOutputPath);
    return fullOutputPath;
}

std::string SaveDataManager::plotPixelOverTime(const std::vector<double>& series,
                                                 const std::string& title,
                                                 const std::string& yLabel,
                                                 const std::string& outputFilename)
{
    if (series.empty()) {
        Logger::get()->error("plotPixelOverTime: series is empty");
        return "";
    }

    ensureDirectoryExists(resultsDir);
    std::string outputPath = (fs::path(resultsDir) / outputFilename).string();

    try {
        int width = 1000;
        int height = 600;
        int margin = 80;
        int plotWidth = width - 2 * margin;
        int plotHeight = height - 2 * margin;

        cv::Mat plot = cv::Mat::ones(height, width, CV_8UC3) * 255; // White background

        // Find min/max for scaling
        double minVal = *std::min_element(series.begin(), series.end());
        double maxVal = *std::max_element(series.begin(), series.end());
        double range = maxVal - minVal;
        if (range == 0) range = 1.0;

        // Draw axes
        cv::line(plot, cv::Point(margin, margin), 
                 cv::Point(margin, height - margin), cv::Scalar(0, 0, 0), 2);
        cv::line(plot, cv::Point(margin, height - margin), 
                 cv::Point(width - margin, height - margin), cv::Scalar(0, 0, 0), 2);

        // Draw the line plot
        std::vector<cv::Point> points;
        for (size_t i = 0; i < series.size(); ++i) {
            int x = margin + static_cast<int>((static_cast<double>(i) / (series.size() - 1)) * plotWidth);
            int y = height - margin - static_cast<int>(((series[i] - minVal) / range) * plotHeight);
            points.push_back(cv::Point(x, y));
        }

        // Draw lines connecting points
        for (size_t i = 1; i < points.size(); ++i) {
            cv::line(plot, points[i - 1], points[i], cv::Scalar(0, 0, 255), 2); // Blue line
        }

        // Draw points
        for (const auto& pt : points) {
            cv::circle(plot, pt, 3, cv::Scalar(0, 0, 255), -1);
        }

        // Add title
        cv::Size titleSize = cv::getTextSize(title, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, nullptr);
        int titleX = (width - titleSize.width) / 2;
        putText(plot, title, cv::Point(titleX, 40), 0.8, cv::Scalar(0, 0, 0));

        // Add axis labels
        putText(plot, "Time Frame", cv::Point(width / 2 - 50, height - 20), 0.6, cv::Scalar(0, 0, 0));
        
        // Y-axis label (rotated)
        cv::Point yLabelPos(margin / 2, height / 2);
        // Note: OpenCV doesn't support rotated text easily, so we'll place it below title
        putText(plot, yLabel, cv::Point(20, height / 2), 0.6, cv::Scalar(0, 0, 0));

        // Draw grid
        for (int i = 0; i <= 10; ++i) {
            int y = margin + (plotHeight * i / 10);
            cv::line(plot, cv::Point(margin, y), cv::Point(width - margin, y), 
                     cv::Scalar(200, 200, 200), 1);
        }

        cv::imwrite(outputPath, plot);
        Logger::get()->info("Series plot saved: " + outputPath);
        return outputPath;
    } catch (const cv::Exception& e) {
        Logger::get()->error("plotPixelOverTime: failed to save plot: " + std::string(e.what()));
        return "";
    }
}

std::string SaveDataManager::plotMask(const cv::Mat& mask,
                                      const std::string& outputFilename)
{
    if (mask.empty()) {
        Logger::get()->error("plotMask: mask is empty");
        return "";
    }

    if (mask.dims != 2) {
        Logger::get()->error("plotMask: mask must be a 2D array");
        return "";
    }

    ensureDirectoryExists(resultsDir);
    std::string outputFile = (fs::path(resultsDir) / outputFilename).string();

    try {
        // Convert mask to displayable format (invert for grayscale display)
        cv::Mat maskToPlot;
        if (mask.type() == CV_8UC1) {
            maskToPlot = (1 - mask) * 255; // Invert: 0 becomes 255, 1 becomes 0
        } else {
            mask.convertTo(maskToPlot, CV_8UC1);
        }

        // Convert to BGR for saving (grayscale visualization)
        cv::Mat colored;
        cv::cvtColor(maskToPlot, colored, cv::COLOR_GRAY2BGR);

        cv::imwrite(outputFile, colored);
        Logger::get()->info("Mask plot saved to: " + outputFile);
        return outputFile;
    } catch (const cv::Exception& e) {
        Logger::get()->error("plotMask: failed to save plot: " + std::string(e.what()));
        return "";
    }
}

