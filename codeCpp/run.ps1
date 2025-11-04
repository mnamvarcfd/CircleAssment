# Run the application in Docker
Write-Host "Running CodeCpp in Docker..." -ForegroundColor Yellow

# Default directory
$dicomDir = "input_data/DICOM_files"
if ($args.Count -gt 0) {
    $dicomDir = $args[0]
}

Write-Host "Using DICOM directory: $dicomDir" -ForegroundColor Cyan
Write-Host ""

# Run in Docker container - build and run
$dicomPath = $dicomDir -replace '\\', '/'
docker-compose run --rm codecpp-dev bash -c "echo 'Building project (this may take a few minutes)...' && cd /app && rm -rf build && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j4 && echo '' && echo '=== Running Application ===' && ./CodeCpp /app/$dicomPath"

