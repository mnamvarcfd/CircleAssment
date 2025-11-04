#include <iostream>
#include <cstring>
#include "logger.h"
#include "DicomLoader.h"
#include "TiffMaskLoader.h"
#include "ComputeQuantity.h"

#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#include <dcmtk/dcmimgle/dcmimage.h>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
    // Determine directory path
    std::string directoryPath;
    if (argc >= 2) {
        directoryPath = argv[1];
    } else {
        directoryPath = "input_data/DICOM_files";
        std::cout << "No directory specified, using default: " << directoryPath << std::endl;
    }

    DicomLoader loader;
    loader.loadDirectory(directoryPath);

    const auto& datasets = loader.getDatasets();
    std::cout << "Loaded " << datasets.size() << " DICOM files.\n";

    if (datasets.empty()) {
        std::cerr << "FATAL ERROR: No DICOM files were loaded successfully." << std::endl;
        return 1;
    }

    // Print patient names
    for (auto ds : datasets) {
        if (!ds) {
            std::cerr << "FATAL ERROR: Null dataset pointer encountered." << std::endl;
            return 1;
        }
        
        OFString patientName;
        if (ds->findAndGetOFString(DCM_PatientName, patientName).good()) {
            std::cout << "Patient Name: " << patientName << std::endl;
        }
    }



    // Load TIFF mask file
    // Use absolute path or path relative to working directory
    std::string maskPath = "/app/input_data/AIF_And_Myo_Masks.tiff";
    std::cout << "Attempting to load mask file: " << maskPath << std::endl;
    
    cv::Mat bloodPoolMask, myoMask;
    try {
        TiffMaskLoader maskLoader(maskPath);
        bloodPoolMask = maskLoader.loadMask(0); // first page (blood pool mask)
        
        std::cout << "Blood pool mask type: " << bloodPoolMask.type() << "\n";
        std::cout << "Blood pool mask size: " << bloodPoolMask.rows << "x" << bloodPoolMask.cols << "\n";
        
        // Try to load second page if it exists (myocardium mask)
        try {
            myoMask = maskLoader.loadMask(1);
            std::cout << "Myocardium mask loaded successfully, size: " << myoMask.rows << "x" << myoMask.cols << "\n";
        } catch (const std::out_of_range& e) {
            std::cerr << "FATAL ERROR: Myocardium mask (second page) not found in TIFF file." << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: Failed to load mask file: " << e.what() << std::endl;
        return 1;
    }

    // Convert DICOM datasets to OpenCV Mat frames
    std::cout << "\nConverting DICOM files to OpenCV Mat frames..." << std::endl;
    std::vector<cv::Mat> frames;
    
    for (size_t i = 0; i < datasets.size(); ++i) {
        DcmDataset* dataset = datasets[i];
        if (!dataset) continue;

        // Create DICOM image object
        DicomImage* dcmImage = new DicomImage(dataset, dataset->getOriginalXfer());
        if (dcmImage == nullptr || dcmImage->getStatus() != EIS_Normal) {
            std::cerr << "Warning: Failed to create DicomImage for dataset " << i << std::endl;
            continue;
        }

        // Get image dimensions
        unsigned int width = dcmImage->getWidth();
        unsigned int height = dcmImage->getHeight();
        unsigned int depth = dcmImage->getDepth();

        // Convert to 8-bit grayscale
        if (dcmImage->getPhotometricInterpretation() == EPI_Monochrome2) {
            dcmImage->setMinMaxWindow();
        }
        dcmImage->setWindow(0, 0); // Use default window
        dcmImage->setPolarity(EPP_Normal);

        // Create OpenCV Mat
        cv::Mat frame(height, width, CV_8UC1);
        
        // Get pixel data
        const void* pixelData = dcmImage->getOutputData(depth);
        if (pixelData) {
            memcpy(frame.data, pixelData, height * width * sizeof(uchar));
        } else {
            std::cerr << "Warning: Failed to get pixel data for dataset " << i << std::endl;
            delete dcmImage;
            continue;
        }

        frames.push_back(frame);
        delete dcmImage;
    }

    std::cout << "Converted " << frames.size() << " DICOM files to frames." << std::endl;

    if (frames.empty()) {
        std::cerr << "FATAL ERROR: No frames were converted from DICOM files." << std::endl;
        return 1;
    }

    // Use ComputeQuantity
    std::cout << "\n=== Computing Quantities ===" << std::endl;
    try {
        ComputeQuantity computeQuantity(frames, bloodPoolMask, myoMask);

        // Compute arterial input function
        std::cout << "Computing arterial input function (AIF)..." << std::endl;
        cv::Mat aif = computeQuantity.arterialInputFunction();
        std::cout << "AIF computed: " << aif.rows << "x" << aif.cols << std::endl;
        std::cout << "AIF type: " << aif.type() << std::endl;

        // Compute myocardium time series
        std::cout << "Computing myocardium time series..." << std::endl;
        auto [myoCoords, myoTimeSeries] = computeQuantity.myocardiumTimeSeries();
        std::cout << "Myocardium time series: " << myoTimeSeries.rows << " pixels x " << myoTimeSeries.cols << " time points" << std::endl;

        // Compute blood pool time series
        std::cout << "Computing blood pool time series..." << std::endl;
        auto [bloodCoords, bloodTimeSeries] = computeQuantity.bloodPoolTimeSeries();
        std::cout << "Blood pool time series: " << bloodTimeSeries.rows << " pixels x " << bloodTimeSeries.cols << " time points" << std::endl;

        std::cout << "\nAll computations completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR in ComputeQuantity: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

