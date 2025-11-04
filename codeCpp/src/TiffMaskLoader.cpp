#include "TiffMaskLoader.h"
#include <fstream>

TiffMaskLoader::TiffMaskLoader(const std::string& filename)
    : maskFile(filename)
{
    loadAllPages();
}

void TiffMaskLoader::loadAllPages()
{
    // Check if file exists
    std::ifstream fileCheck(maskFile);
    if (!fileCheck.good()) {
        throw std::runtime_error("TIFF file does not exist or cannot be accessed: " + maskFile);
    }
    fileCheck.close();
    
    // Read all pages from the TIFF file
    bool success = cv::imreadmulti(maskFile, tiffPages, cv::IMREAD_UNCHANGED);
    if (!success) {
        throw std::runtime_error("OpenCV imreadmulti() failed for file: " + maskFile + ". File may be corrupted or format not supported.");
    }
    if (tiffPages.empty()) {
        throw std::runtime_error("TIFF file loaded but contains no pages: " + maskFile);
    }
    std::cout << "Loaded " << tiffPages.size() << " pages from " << maskFile << "\n";
}

cv::Mat TiffMaskLoader::loadMask(int maskIndex)
{
    if (maskIndex < 0 || maskIndex >= static_cast<int>(tiffPages.size())) {
        throw std::out_of_range("Mask index out of range");
    }

    cv::Mat mask = tiffPages[maskIndex].clone();

    // Ensure binary mask (0 and 1)
    cv::threshold(mask, mask, 0, 1, cv::THRESH_BINARY);

    std::cout << "Mask loaded with shape: " << mask.rows << "x" << mask.cols << "\n";

    return mask;
}
