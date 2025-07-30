#!/bin/bash

# SPAdes GPU Docker Test Script
# Tests the built Docker container functionality

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if image exists
IMAGE_NAME="spades:gpu-lite"
if ! docker images | grep -q "$IMAGE_NAME"; then
    print_error "Image $IMAGE_NAME not found. Build it first with:"
    echo "docker build -f docker/Dockerfile.lightweight -t $IMAGE_NAME ."
    exit 1
fi

print_status "Testing SPAdes GPU Docker container..."

# Test 1: Basic container startup
print_status "Test 1: Basic container startup"
if docker run --rm $IMAGE_NAME echo "Container startup test successful"; then
    print_success "Container starts successfully"
else
    print_error "Container failed to start"
    exit 1
fi

# Test 2: SPAdes command availability
print_status "Test 2: SPAdes command availability"
if docker run --rm $IMAGE_NAME which spades.py > /dev/null 2>&1; then
    print_success "SPAdes command found"
else
    print_error "SPAdes command not found"
    exit 1
fi

# Test 3: GPU support check (if available)
print_status "Test 3: GPU support check"
GPU_AVAILABLE=$(docker run --rm --gpus all $IMAGE_NAME nvidia-smi > /dev/null 2>&1 && echo "true" || echo "false")
if [ "$GPU_AVAILABLE" = "true" ]; then
    print_success "GPU is available and accessible"
    docker run --rm --gpus all $IMAGE_NAME nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    print_warning "GPU not available (this is expected on macOS or systems without NVIDIA GPUs)"
fi

# Test 4: Python imports
print_status "Test 4: Python imports test"
if docker run --rm $IMAGE_NAME python3 -c "import numpy, matplotlib; print('Python dependencies OK')"; then
    print_success "Python dependencies working"
else
    print_error "Python dependencies missing"
    exit 1
fi

# Test 5: SPAdes help
print_status "Test 5: SPAdes help command"
if docker run --rm $IMAGE_NAME spades.py --help | head -10 | grep -q "SPAdes"; then
    print_success "SPAdes help command working"
else
    print_error "SPAdes help command failed"
    exit 1
fi

# Test 6: GPU environment variables
print_status "Test 6: GPU environment variables"
docker run --rm $IMAGE_NAME env | grep SPADES_GPU || true

print_success "All tests completed successfully!"
print_status "Container is ready for use."

echo
print_status "Usage examples:"
echo "# Basic usage:"
echo "docker run --rm -v \$(pwd)/data:/data $IMAGE_NAME spades.py --help"
echo
echo "# With GPU support (Linux with NVIDIA GPU):"
echo "docker run --rm --gpus all -v \$(pwd)/data:/data $IMAGE_NAME spades.py --help"
echo
echo "# Interactive shell:"
echo "docker run --rm -it -v \$(pwd)/data:/data $IMAGE_NAME bash"
echo
echo "# Using docker-compose:"
echo "docker-compose -f docker/docker-compose.yml up" 