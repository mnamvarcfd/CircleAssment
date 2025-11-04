#include <gtest/gtest.h>
#include "SaveDataManager.h"
#include "logger.h"
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

class SaveDataManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize logger for tests
        Logger::init("test_logs/app.log");
        
        // Use absolute path to /app/test_results so it's accessible from host via Docker volume
        // This will be mounted to ./test_results on the host
        testResultsDir = "/app/test_results";
        manager = std::make_unique<SaveDataManager>(testResultsDir);
    }

    void TearDown() override {
        // Clean up: remove test results directory after tests
        // COMMENT OUT the following lines if you want to keep test_mask.png
        // if (fs::exists(testResultsDir)) {
        //     fs::remove_all(testResultsDir);
        // }
    }

    std::string testResultsDir;
    std::unique_ptr<SaveDataManager> manager;
};

TEST_F(SaveDataManagerTest, PlotMask_SimpleBinaryMask) {
    // Create a simple binary mask (100x100, with some pixels set to 1)
    cv::Mat mask = cv::Mat::zeros(100, 100, CV_8UC1);
    
    // Create a simple pattern: set a 20x20 square in the center to 1
    cv::Rect centerRect(40, 40, 20, 20);
    mask(centerRect) = 1;

    // Test plotMask
    std::string outputPath = manager->plotMask(mask, "test_mask.png");

    // Verify the function returned a non-empty path
    ASSERT_FALSE(outputPath.empty());

    // Verify the file was created
    std::string expectedPath = (fs::path(testResultsDir) / "test_mask.png").string();
    ASSERT_EQ(outputPath, expectedPath);
    ASSERT_TRUE(fs::exists(expectedPath));

    // Verify the file is not empty
    ASSERT_GT(fs::file_size(expectedPath), 0);

    // Verify we can read the image back
    cv::Mat loadedImage = cv::imread(expectedPath);
    ASSERT_FALSE(loadedImage.empty());
    ASSERT_EQ(loadedImage.rows, 100);
    ASSERT_EQ(loadedImage.cols, 100);
}

TEST_F(SaveDataManagerTest, PlotMask_EmptyMask) {
    // Test with empty mask
    cv::Mat emptyMask;
    
    std::string outputPath = manager->plotMask(emptyMask, "empty_mask.png");

    // Should return empty string for empty mask
    ASSERT_TRUE(outputPath.empty());
}

