#include "DicomLoader.h"
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmdata/dcfilefo.h>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

DicomLoader::~DicomLoader() {
    // Free memory
    for (auto fileFormat : fileFormats) {
        delete fileFormat;
    }
    datasets.clear();
}

void DicomLoader::loadDirectory(const std::string& dirPath) {
    if (!fs::exists(dirPath)) {
        std::cerr << "FATAL ERROR: Directory does not exist: " << dirPath << std::endl;
        return;
    }
    if (!fs::is_directory(dirPath)) {
        std::cerr << "FATAL ERROR: Path is not a directory: " << dirPath << std::endl;
        return;
    }
    
    for (const auto& entry : fs::directory_iterator(dirPath)) {
        if (!entry.is_regular_file()) continue;
        const auto& path = entry.path();
        if (path.extension() == ".dcm") {  // Simple check for .dcm files
            loadFile(path.string());
        }
    }
}

void DicomLoader::loadFile(const std::string& filename) {
    DcmFileFormat* fileFormat = new DcmFileFormat();
    OFCondition status = fileFormat->loadFile(filename.c_str());
    if (!status.good()) {
        std::cerr << "Failed to load " << filename << ": " << status.text() << std::endl;
        delete fileFormat;
        return;
    }

    // Store the dataset pointer
    DcmDataset* dataset = fileFormat->getDataset();
    if (!dataset) {
        std::cerr << "FATAL ERROR: getDataset() returned nullptr for " << filename << std::endl;
        delete fileFormat;
        return;
    }
    
    datasets.push_back(dataset);
    fileFormats.push_back(fileFormat);  // Keep fileFormat alive
}

const std::vector<DcmDataset*>& DicomLoader::getDatasets() const {
    return datasets;
}
