# ===========================
# Stage 1: Build dependencies and compile the project
# ===========================
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build tools and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    pkg-config \
    zlib1g-dev \
    libtiff-dev \
    libpng-dev \
    libxml2-dev \
    libssl-dev \
    libopencv-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libdcmtk-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ===========================
# Download and install ALGLIB
# ===========================
WORKDIR /opt

RUN wget -q https://www.alglib.net/translator/re/alglib-3.20.0.cpp.gpl.tgz && \
    tar -xzf alglib-3.20.0.cpp.gpl.tgz && \
    cd alglib-cpp && \
    mkdir -p /usr/local/include/alglib && \
    cp src/*.h /usr/local/include/alglib/ && \
    cp src/*.cpp /usr/local/include/alglib/ && \
    rm -rf /opt/alglib-*

# ===========================
# Build your C++ project
# ===========================
WORKDIR /app

COPY . .

RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . -j$(nproc)

# ===========================
# Stage 2: Runtime (lightweight)
# ===========================
FROM ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libdcmtk16 \
    zlib1g \
    libtiff5 \
    libpng16-16 \
    libxml2 \
    libssl3 \
    libopencv-core4.5 \
    libopencv-imgproc4.5 \
    libopencv-imgcodecs4.5 \
    libeigen3-dev \
    libsuitesparse-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy ALGLIB headers and compiled project binary
COPY --from=builder /usr/local/include/alglib /usr/local/include/alglib
COPY --from=builder /app/build/CodeCpp /usr/local/bin/CodeCpp

# Working directory and I/O folders
WORKDIR /app
RUN mkdir -p /app/input_data /app/logs /app/output

# Default command
CMD ["/usr/local/bin/CodeCpp"]
