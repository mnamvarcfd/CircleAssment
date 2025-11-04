#include <iostream>
#include "logger.h"
#include "DicomLoader.h"
#include "TiffMaskLoader.h"

#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dcdeftag.h>

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
    
    try {
        TiffMaskLoader maskLoader(maskPath);
        cv::Mat mask0 = maskLoader.loadMask(0); // first page
        
        std::cout << "Mask0 type: " << mask0.type() << "\n";
        std::cout << "Mask0 size: " << mask0.rows << "x" << mask0.cols << "\n";
        
        // Try to load second page if it exists
        try {
            cv::Mat mask1 = maskLoader.loadMask(1);
            std::cout << "Mask1 loaded successfully, size: " << mask1.rows << "x" << mask1.cols << "\n";
        } catch (const std::out_of_range& e) {
            std::cout << "Only one mask page available (mask1 doesn't exist)\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: Failed to load mask file: " << e.what() << std::endl;
        return 1;
    }


    return 0;
}

