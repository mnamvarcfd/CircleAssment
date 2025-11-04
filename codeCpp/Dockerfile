# Multi-stage Dockerfile for C++ DICOM project
FROM ubuntu:22.04 AS builder

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build essentials and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    pkg-config \
    libdcmtk-dev \
    libdcmtk16 \
    zlib1g-dev \
    libtiff-dev \
    libpng-dev \
    libxml2-dev \
    libssl-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Create build directory
RUN mkdir -p build

# Build the project
WORKDIR /app/build
RUN cmake .. && \
    cmake --build . -j$(nproc)

# Runtime stage
FROM ubuntu:22.04 AS runtime

# Install runtime dependencies
# Note: Installing libopencv-dev for runtime to ensure all OpenCV libraries are available
# In production, you could install only the specific runtime packages, but this is simpler
RUN apt-get update && apt-get install -y \
    libdcmtk16 \
    zlib1g \
    libtiff5 \
    libpng16-16 \
    libxml2 \
    libssl3 \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy built executable from builder stage
# Note: executable is in build/ directory (not build/bin/ due to CMAKE_RUNTIME_OUTPUT_DIRECTORY setting)
COPY --from=builder /app/build/CodeCpp /usr/local/bin/CodeCpp

# Set working directory
WORKDIR /app

# Create directories for input/output
RUN mkdir -p /app/input_data /app/logs /app/output

# Default command
CMD ["/usr/local/bin/CodeCpp"]

