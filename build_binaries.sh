#!/bin/bash

set -e

CUDA_VERSIONS=("11.8.0" "12.6.3" "12.8.1")
PYTHON_VERSIONS=("3.10" "3.11" "3.12")
OUTPUT_DIR="gKDTree/precompiled_binaries"

mkdir -p $OUTPUT_DIR

for cuda_ver in "${CUDA_VERSIONS[@]}"; do
    for py_ver in "${PYTHON_VERSIONS[@]}"; do
        echo "Building for CUDA ${cuda_ver}, Python ${py_ver}"
        
        # Create Dockerfile for this combination
        cat > Dockerfile.build <<EOF
ARG BASE_IMAGE=docker.io/nvidia/cuda:${cuda_ver}-devel-ubuntu22.04
FROM --platform=linux/amd64 \$BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \\
    software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa \\
    && apt update && apt install -y \\
    python${py_ver} python${py_ver}-venv python${py_ver}-dev \\
    build-essential curl \\
    && rm -rf /var/lib/apt/lists/*

# Install pip for this Python version
RUN curl https://bootstrap.pypa.io/get-pip.py | python${py_ver}
RUN python${py_ver} -m pip install pybind11 cmake==3.24

WORKDIR /workspace
COPY . .

RUN mkdir -p build && cd build && \\
    cmake .. -DCMAKE_BUILD_TYPE=Release \\
    -DPYTHON_EXECUTABLE=\$(which python${py_ver}) && \\
    make -j\$(nproc)
EOF
        
        docker build -f Dockerfile.build -t gkdtree-build:cuda${cuda_ver}-py${py-ver} .

        container_id=$(docker container create gkdtree-build:cuda${cuda_ver}-py${py-ver})

        docker cp ${container_id}:/workspace/build/ temp_build

        cuda_ver_short=${cuda_ver%.*}

        find ./temp_build/ -type f -name "_internal*.so" | xargs -I {} cp {} ${OUTPUT_DIR}/_internal_cuda${cuda_ver_short//./_}-py${py_ver//./_}_$(arch).so

        docker rm ${container_id}
        rm -rf temp_build
        rm Dockerfile.build

    done
done