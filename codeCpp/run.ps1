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
docker-compose run --rm codecpp-dev bash -c "cd /app/build && if [ ! -f CodeCpp ]; then echo 'Building project (this may take a few minutes)...' && cd .. && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j4; else echo 'Using existing build...'; fi && cd /app/build && echo '' && echo '=== Running Application ===' && ./CodeCpp /app/$dicomPath"

