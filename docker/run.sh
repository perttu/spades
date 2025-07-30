#!/bin/bash

# SPAdes GPU Docker Run Script
# Quick setup and run script for Docker Compose

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

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_status "SPAdes GPU Docker Setup"
print_status "Project root: $PROJECT_ROOT"

# Change to project root
cd "$PROJECT_ROOT"

# Create data directories if they don't exist
print_status "Creating data directories..."
mkdir -p docker/data/{input,output,tmp,config,logs}

# Check if Docker is running
if ! docker info &> /dev/null; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check for NVIDIA Docker support
if docker run --rm --gpus all nvidia/cuda:12.4.1-base nvidia-smi &> /dev/null; then
    print_success "NVIDIA Docker runtime detected"
else
    print_warning "NVIDIA Docker runtime not available - GPU features will not work"
fi

# Check if image exists, build if not
if ! docker image inspect spades:latest &> /dev/null; then
    print_status "SPAdes image not found. Building..."
    if ! docker/build.sh; then
        print_error "Failed to build SPAdes image"
        exit 1
    fi
else
    print_status "SPAdes image found"
fi

# Start services
print_status "Starting SPAdes GPU services..."
if docker-compose -f docker/docker-compose.yml up -d; then
    print_success "Services started successfully!"
else
    print_error "Failed to start services"
    exit 1
fi

# Wait a moment for services to start
sleep 3

# Check service status
print_status "Checking service status..."
docker-compose -f docker/docker-compose.yml ps

# Show usage examples
echo
print_success "SPAdes GPU is now running! Usage examples:"
echo
echo "# Check GPU access:"
echo "docker-compose -f docker/docker-compose.yml exec spades-gpu nvidia-smi"
echo
echo "# Interactive shell:"
echo "docker-compose -f docker/docker-compose.yml exec spades-gpu bash"
echo
echo "# Run SPAdes assembly:"
echo "docker-compose -f docker/docker-compose.yml exec spades-gpu \\"
echo "    spades.py -1 /data/input/reads_1.fastq -2 /data/input/reads_2.fastq -o /data/output/assembly --gpu"
echo
echo "# View logs:"
echo "docker-compose -f docker/docker-compose.yml logs -f spades-gpu"
echo
echo "# Stop services:"
echo "docker-compose -f docker/docker-compose.yml down"
echo
print_status "Data directories are mounted at:"
print_status "  Input: $(pwd)/docker/data/input"
print_status "  Output: $(pwd)/docker/data/output"
print_status "  Temp: $(pwd)/docker/data/tmp"
print_status "  Config: $(pwd)/docker/data/config"
print_status "  Logs: $(pwd)/docker/data/logs" 