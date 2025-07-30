#!/bin/bash

# SPAdes GPU Docker Build Script
# Usage: ./build.sh [options]

set -e

# Default values
BUILD_TYPE="Release"
GPU_SUPPORT="ON"
CUDA_SUPPORT="ON"
TARGET="production"
TAG="latest"
PLATFORM="linux/amd64"
CACHE="true"
PUSH="false"
REGISTRY=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
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

# Print usage information
usage() {
    cat << EOF
SPAdes GPU Docker Build Script

Usage: $0 [options]

Options:
    -t, --target TARGET         Build target (production, builder, jupyter) [default: production]
    -g, --tag TAG              Docker image tag [default: latest]
    -b, --build-type TYPE      CMake build type (Release, Debug, RelWithDebInfo) [default: Release]
    --no-gpu                   Disable GPU support
    --no-cuda                  Disable CUDA support
    --platform PLATFORM       Target platform [default: linux/amd64]
    --no-cache                 Disable build cache
    --push                     Push image to registry after build
    --registry REGISTRY        Docker registry for push
    -j, --jupyter              Build Jupyter image
    -m, --monitoring           Build with monitoring tools
    -d, --dev                  Development build (Debug + additional tools)
    -c, --clean                Clean build (remove existing images)
    -v, --verbose              Verbose output
    -h, --help                 Show this help message

Examples:
    $0                                    # Basic production build
    $0 --target builder --build-type Debug --tag dev
    $0 --jupyter --tag jupyter-latest
    $0 --dev --no-cache
    $0 --push --registry myregistry.com/spades

Environment Variables:
    DOCKER_BUILDKIT=1          Enable BuildKit (recommended)
    CUDA_VERSION               Override CUDA version (default: 12.3)
    CMAKE_BUILD_TYPE           Override build type
    SPADES_VERSION             Override SPAdes version tag

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        -g|--tag)
            TAG="$2"
            shift 2
            ;;
        -b|--build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --no-gpu)
            GPU_SUPPORT="OFF"
            shift
            ;;
        --no-cuda)
            CUDA_SUPPORT="OFF"
            shift
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --no-cache)
            CACHE="false"
            shift
            ;;
        --push)
            PUSH="true"
            shift
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        -j|--jupyter)
            TARGET="jupyter"
            TAG="jupyter-${TAG}"
            shift
            ;;
        -m|--monitoring)
            # Add monitoring tools to build
            BUILD_ARGS="$BUILD_ARGS --build-arg INSTALL_MONITORING=true"
            shift
            ;;
        -d|--dev)
            BUILD_TYPE="Debug"
            TAG="dev-${TAG}"
            TARGET="builder"
            shift
            ;;
        -c|--clean)
            print_status "Cleaning existing images..."
            docker image prune -f
            docker rmi -f spades:$TAG 2>/dev/null || true
            shift
            ;;
        -v|--verbose)
            set -x
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate target
case $TARGET in
    production|builder|jupyter)
        ;;
    *)
        print_error "Invalid target: $TARGET. Must be one of: production, builder, jupyter"
        exit 1
        ;;
esac

# Set image name
if [[ -n "$REGISTRY" ]]; then
    IMAGE_NAME="${REGISTRY}/spades:${TAG}"
else
    IMAGE_NAME="spades:${TAG}"
fi

# Detect project root (go up from docker/ directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_status "Building SPAdes Docker image..."
print_status "Project root: $PROJECT_ROOT"
print_status "Target: $TARGET"
print_status "Image name: $IMAGE_NAME"
print_status "Build type: $BUILD_TYPE"
print_status "GPU support: $GPU_SUPPORT"
print_status "CUDA support: $CUDA_SUPPORT"
print_status "Platform: $PLATFORM"

# Check prerequisites
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check for NVIDIA Docker support if GPU is enabled
if [[ "$GPU_SUPPORT" == "ON" ]]; then
    if ! docker run --rm --gpus all nvidia/cuda:12.3-base nvidia-smi &> /dev/null; then
        print_warning "GPU support requested but NVIDIA Docker runtime not available"
        print_warning "Continuing with build, but GPU features may not work"
    fi
fi

# Set up build arguments
BUILD_ARGS=""
BUILD_ARGS="$BUILD_ARGS --build-arg CMAKE_BUILD_TYPE=$BUILD_TYPE"
BUILD_ARGS="$BUILD_ARGS --build-arg SPADES_GPU_SUPPORT=$GPU_SUPPORT"
BUILD_ARGS="$BUILD_ARGS --build-arg SPADES_CUDA_SUPPORT=$CUDA_SUPPORT"
BUILD_ARGS="$BUILD_ARGS --platform $PLATFORM"
BUILD_ARGS="$BUILD_ARGS --target $TARGET"

# Add cache options
if [[ "$CACHE" == "true" ]]; then
    BUILD_ARGS="$BUILD_ARGS --build-arg BUILDKIT_INLINE_CACHE=1"
else
    BUILD_ARGS="$BUILD_ARGS --no-cache"
fi

# Choose Dockerfile
DOCKERFILE="$SCRIPT_DIR/Dockerfile"
if [[ "$TARGET" == "jupyter" ]]; then
    DOCKERFILE="$SCRIPT_DIR/Dockerfile.jupyter"
fi

# Build command
BUILD_CMD="docker build"
BUILD_CMD="$BUILD_CMD -f $DOCKERFILE"
BUILD_CMD="$BUILD_CMD -t $IMAGE_NAME"
BUILD_CMD="$BUILD_CMD $BUILD_ARGS"
BUILD_CMD="$BUILD_CMD $PROJECT_ROOT"

print_status "Executing: $BUILD_CMD"

# Execute build
if eval $BUILD_CMD; then
    print_success "Build completed successfully!"
    
    # Show image size
    IMAGE_SIZE=$(docker images --format "table {{.Size}}" $IMAGE_NAME | tail -n +2)
    print_status "Image size: $IMAGE_SIZE"
    
    # Push if requested
    if [[ "$PUSH" == "true" ]]; then
        print_status "Pushing image to registry..."
        if docker push $IMAGE_NAME; then
            print_success "Image pushed successfully!"
        else
            print_error "Failed to push image"
            exit 1
        fi
    fi
    
    # Show usage instructions
    echo
    print_success "Build complete! Usage examples:"
    echo
    echo "# Run interactive shell:"
    echo "docker run --rm -it --gpus all -v \$(pwd)/data:/data $IMAGE_NAME bash"
    echo
    echo "# Run SPAdes assembly:"
    echo "docker run --rm --gpus all -v \$(pwd)/data:/data $IMAGE_NAME \\"
    echo "    spades.py -1 /data/input/reads_1.fastq -2 /data/input/reads_2.fastq -o /data/output"
    echo
    echo "# Use with docker-compose:"
    echo "docker-compose -f docker/docker-compose.yml up"
    
    if [[ "$TARGET" == "jupyter" ]]; then
        echo
        echo "# Start Jupyter lab:"
        echo "docker run --rm -p 8888:8888 --gpus all $IMAGE_NAME"
        echo "# Access at: http://localhost:8888 (token: spades-gpu-token)"
    fi
    
else
    print_error "Build failed!"
    exit 1
fi 