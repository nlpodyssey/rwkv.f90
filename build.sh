#!/bin/bash

BUILD_DIR="build"
OUTPUT_DIR="output"

if [ -d "${BUILD_DIR}" ]; then
    echo "Removing existing ${BUILD_DIR} directory."
    rm -rf "${BUILD_DIR}"
fi

if [ ! -d "${OUTPUT_DIR}/bin" ]; then
    mkdir -p "${OUTPUT_DIR}/bin"
fi

if [ ! -d "${OUTPUT_DIR}/lib" ]; then
    mkdir -p "${OUTPUT_DIR}/lib"
fi

echo "Creating new ${BUILD_DIR} directory."
mkdir "${BUILD_DIR}"

echo "Changing directory to ${BUILD_DIR}."
cd "${BUILD_DIR}" || { echo "Failed to change directory. Exiting."; exit 1; }

echo "Running CMake..."
if ! cmake -DBLAS_LIBRARY=${DBLAS_LIBRARY} ..; then
    echo "CMake failed. Exiting."
    exit 1
fi

echo "Running make..."
if ! make; then
    echo "Make failed. Exiting."
    exit 1
fi

echo "Moving executable to ../${OUTPUT_DIR}/bin/"
mv rwkv "../${OUTPUT_DIR}/bin/"

echo "Moving library to ../${OUTPUT_DIR}/lib/"
mv librwkv_lib.a "../${OUTPUT_DIR}/lib/"

echo "Cleaning up temporary files..."
if ! make clean; then
    echo "Clean failed. Exiting."
    exit 1
fi

echo "Build completed successfully."
