#pragma once

#include <vector>
#include <string>

// Forward declarations - DCMTK includes are in .cpp file to avoid header dependency issues
class DcmDataset;
class DcmFileFormat;

class DicomLoader {
public:
    DicomLoader() = default;
    ~DicomLoader();

    // Load all DICOM files from a directory
    void loadDirectory(const std::string& dirPath);

    // Access loaded datasets
    const std::vector<DcmDataset*>& getDatasets() const;

private:
    std::vector<DcmDataset*> datasets;         // Pointers to DICOM datasets
    std::vector<DcmFileFormat*> fileFormats;   // Keep file formats alive

    // Load a single file
    void loadFile(const std::string& filename);
};
