//***************************************************************************
//* Copyright (c) 2023-2024 SPAdes team
//* Copyright (c) 2020-2022 Saint Petersburg State University
//* All Rights Reserved
//* See file LICENSE for details.
//***************************************************************************

#pragma once

#ifdef SPADES_GPU_SUPPORT

#include "gpu_device.hpp"
#include <memory>
#include <vector>
#include <queue>
#include <mutex>

#ifdef SPADES_CUDA_SUPPORT
#include <cuda_runtime.h>
#endif

namespace gpu {

// GPU memory deleter for smart pointers
struct GPUDeleter {
    void operator()(void* ptr) const {
#ifdef SPADES_CUDA_SUPPORT
        if (ptr) {
            cudaFree(ptr);
        }
#endif
    }
};

// Smart pointer for GPU memory
template<typename T>
using gpu_unique_ptr = std::unique_ptr<T, GPUDeleter>;

// GPU memory allocator
class GPUMemoryAllocator {
public:
    static void* allocate(size_t bytes);
    static void deallocate(void* ptr);
    
    template<typename T>
    static gpu_unique_ptr<T> allocate_array(size_t count) {
        size_t bytes = count * sizeof(T);
        T* ptr = static_cast<T*>(allocate(bytes));
        return gpu_unique_ptr<T>(ptr);
    }
    
    template<typename T>
    static gpu_unique_ptr<T> allocate_single() {
        return allocate_array<T>(1);
    }
};

// Memory pool for frequent allocations
class GPUMemoryPool {
public:
    GPUMemoryPool(size_t block_size, size_t max_blocks = 16);
    ~GPUMemoryPool();
    
    void* allocate(size_t bytes);
    void deallocate(void* ptr);
    void clear();
    
    size_t get_allocated_bytes() const { return allocated_bytes_; }
    size_t get_available_blocks() const { return available_blocks_.size(); }

private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
    };
    
    size_t block_size_;
    size_t max_blocks_;
    size_t allocated_bytes_;
    std::vector<MemoryBlock> blocks_;
    std::queue<size_t> available_blocks_;
    std::mutex mutex_;
    
    void allocate_new_block();
};

// Host-device memory transfer utilities
class GPUTransfer {
public:
    // Synchronous transfers
    template<typename T>
    static void host_to_device(const T* host_ptr, T* device_ptr, size_t count) {
#ifdef SPADES_CUDA_SUPPORT
        GPU_CHECK(cudaMemcpy(device_ptr, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
#endif
    }
    
    template<typename T>
    static void device_to_host(const T* device_ptr, T* host_ptr, size_t count) {
#ifdef SPADES_CUDA_SUPPORT
        GPU_CHECK(cudaMemcpy(host_ptr, device_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
#endif
    }
    
    template<typename T>
    static void device_to_device(const T* src_ptr, T* dst_ptr, size_t count) {
#ifdef SPADES_CUDA_SUPPORT
        GPU_CHECK(cudaMemcpy(dst_ptr, src_ptr, count * sizeof(T), cudaMemcpyDeviceToDevice));
#endif
    }
    
    // Asynchronous transfers
    template<typename T>
    static void host_to_device_async(const T* host_ptr, T* device_ptr, size_t count, cudaStream_t stream = 0) {
#ifdef SPADES_CUDA_SUPPORT
        GPU_CHECK(cudaMemcpyAsync(device_ptr, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice, stream));
#endif
    }
    
    template<typename T>
    static void device_to_host_async(const T* device_ptr, T* host_ptr, size_t count, cudaStream_t stream = 0) {
#ifdef SPADES_CUDA_SUPPORT
        GPU_CHECK(cudaMemcpyAsync(host_ptr, device_ptr, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
#endif
    }
    
    // Memory set operations
    template<typename T>
    static void memset_device(T* device_ptr, int value, size_t count) {
#ifdef SPADES_CUDA_SUPPORT
        GPU_CHECK(cudaMemset(device_ptr, value, count * sizeof(T)));
#endif
    }
};

// Pinned host memory for faster transfers
class PinnedMemory {
public:
    template<typename T>
    static T* allocate(size_t count) {
#ifdef SPADES_CUDA_SUPPORT
        T* ptr;
        GPU_CHECK(cudaMallocHost(&ptr, count * sizeof(T)));
        return ptr;
#else
        return nullptr;
#endif
    }
    
    static void deallocate(void* ptr) {
#ifdef SPADES_CUDA_SUPPORT
        if (ptr) {
            cudaFreeHost(ptr);
        }
#endif
    }
};

// Unified memory allocator (if supported)
class UnifiedMemory {
public:
    template<typename T>
    static T* allocate(size_t count) {
#ifdef SPADES_CUDA_SUPPORT
        if (GPUDevice::instance().has_unified_memory()) {
            T* ptr;
            GPU_CHECK(cudaMallocManaged(&ptr, count * sizeof(T)));
            return ptr;
        }
#endif
        return nullptr;
    }
    
    static void deallocate(void* ptr) {
#ifdef SPADES_CUDA_SUPPORT
        if (ptr) {
            cudaFree(ptr);
        }
#endif
    }
    
    static void prefetch_to_device(void* ptr, size_t bytes, int device_id = -1) {
#ifdef SPADES_CUDA_SUPPORT
        if (device_id < 0) {
            device_id = GPUDevice::instance().get_current_device();
        }
        GPU_CHECK(cudaMemPrefetchAsync(ptr, bytes, device_id));
#endif
    }
    
    static void prefetch_to_host(void* ptr, size_t bytes) {
#ifdef SPADES_CUDA_SUPPORT
        GPU_CHECK(cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId));
#endif
    }
};

// GPU memory statistics
class GPUMemoryStats {
public:
    struct Stats {
        size_t total_allocations;
        size_t total_deallocations;
        size_t current_usage;
        size_t peak_usage;
        size_t total_bytes_allocated;
        size_t total_bytes_deallocated;
    };
    
    static void record_allocation(size_t bytes);
    static void record_deallocation(size_t bytes);
    static Stats get_stats();
    static void reset_stats();
    static void print_stats();

private:
    static Stats stats_;
    static std::mutex stats_mutex_;
};

// RAII wrapper for GPU streams
#ifdef SPADES_CUDA_SUPPORT
class GPUStream {
public:
    GPUStream() {
        GPU_CHECK(cudaStreamCreate(&stream_));
    }
    
    ~GPUStream() {
        cudaStreamDestroy(stream_);
    }
    
    cudaStream_t get() const { return stream_; }
    
    void synchronize() {
        GPU_CHECK(cudaStreamSynchronize(stream_));
    }
    
    bool is_ready() const {
        return cudaStreamQuery(stream_) == cudaSuccess;
    }

private:
    cudaStream_t stream_;
    
    // Disable copy
    GPUStream(const GPUStream&) = delete;
    GPUStream& operator=(const GPUStream&) = delete;
};
#endif

} // namespace gpu

#else // SPADES_GPU_SUPPORT not defined

namespace gpu {

// Dummy implementations when GPU support is disabled
struct GPUDeleter {
    void operator()(void*) const {}
};

template<typename T>
using gpu_unique_ptr = std::unique_ptr<T, GPUDeleter>;

class GPUMemoryAllocator {
public:
    static void* allocate(size_t) { return nullptr; }
    static void deallocate(void*) {}
    
    template<typename T>
    static gpu_unique_ptr<T> allocate_array(size_t) {
        return gpu_unique_ptr<T>(nullptr);
    }
    
    template<typename T>
    static gpu_unique_ptr<T> allocate_single() {
        return gpu_unique_ptr<T>(nullptr);
    }
};

class GPUMemoryPool {
public:
    GPUMemoryPool(size_t, size_t = 16) {}
    void* allocate(size_t) { return nullptr; }
    void deallocate(void*) {}
    void clear() {}
    size_t get_allocated_bytes() const { return 0; }
    size_t get_available_blocks() const { return 0; }
};

} // namespace gpu

#endif // SPADES_GPU_SUPPORT 