# Clean rebuild script for Windows
Write-Host "Cleaning build directory..." -ForegroundColor Yellow
if (Test-Path build) {
    # Close any processes that might be using the files
    Get-Process | Where-Object { $_.Path -like "*codeCpp*" -or $_.Path -like "*CodeCpp*" } | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500
    
    # Remove _deps directory first (where FetchContent stores dependencies)
    if (Test-Path build\_deps) {
        Write-Host "Removing dependencies directory..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force build\_deps -ErrorAction SilentlyContinue
    }
    
    # Try to remove with retries in case files are locked
    $maxRetries = 3
    $retryCount = 0
    $removed = $false
    
    while (-not $removed -and $retryCount -lt $maxRetries) {
        try {
            Remove-Item -Recurse -Force build -ErrorAction Stop
            $removed = $true
        } catch {
            $retryCount++
            if ($retryCount -lt $maxRetries) {
                Write-Host "Retry ${retryCount}/${maxRetries}: Some files may be locked, waiting..." -ForegroundColor Yellow
                Start-Sleep -Seconds 1
            } else {
                Write-Host "Warning: Could not fully remove build directory. Some files may be in use." -ForegroundColor Yellow
                Write-Host "Attempting to continue anyway..." -ForegroundColor Yellow
            }
        }
    }
}
Write-Host "Creating build directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path build | Out-Null
Write-Host "Configuring CMake..." -ForegroundColor Yellow
cd build
$cmakeResult = cmake .. 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed!" -ForegroundColor Red
    Write-Host $cmakeResult
    cd ..
    exit 1
}
Write-Host "Building project..." -ForegroundColor Yellow
$buildResult = cmake --build . 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    Write-Host $buildResult
    cd ..
    exit 1
}
Write-Host "Build complete!" -ForegroundColor Green
cd ..
Write-Host "Running executable..." -ForegroundColor Yellow
if (Test-Path .\build\Debug\CodeCpp.exe) {
    .\build\Debug\CodeCpp.exe
} else {
    Write-Host "Executable not found at .\build\Debug\CodeCpp.exe" -ForegroundColor Yellow
    Write-Host "Build may have succeeded but executable is in a different location." -ForegroundColor Yellow
}

