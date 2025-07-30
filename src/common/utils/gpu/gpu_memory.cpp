//***************************************************************************
//* Copyright (c) 2023-2024 SPAdes team
//* Copyright (c) 2020-2022 Saint Petersburg State University
//* All Rights Reserved
//* See file LICENSE for details.
//***************************************************************************

#include "gpu_memory.hpp"
#include "utils/logger/logger.hpp"

#ifdef SPADES_GPU_SUPPORT

#include <algorithm>
#include <iomanip>

namespace gpu {

// GPUMemoryAllocator implementation
void* GPUMemoryAllocator::allocate(size_t bytes) {
#ifdef SPADES_CUDA_SUPPORT
    void* ptr;
    GPU_CHECK(cudaMalloc(&ptr, bytes));
    GPUMemoryStats::record_allocation(bytes);
    return ptr;
#else
    return nullptr;
#endif
}

void GPUMemoryAllocator::deallocate(void* ptr) {
#ifdef SPADES_CUDA_SUPPORT
    if (ptr) {
        // We can't easily track deallocation size here
        // This is a limitation of the CUDA API
        GPUMemoryStats::record_deallocation(0);
        cudaFree(ptr);
    }
#endif
}

// GPUMemoryPool implementation
GPUMemoryPool::GPUMemoryPool(size_t block_size, size_t max_blocks)
    : block_size_(block_size), max_blocks_(max_blocks), allocated_bytes_(0) {
    blocks_.reserve(max_blocks);
}

GPUMemoryPool::~GPUMemoryPool() {
    clear();
}

void* GPUMemoryPool::allocate(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (bytes > block_size_) {
        // Allocate directly for large requests
        return GPUMemoryAllocator::allocate(bytes);
    }
    
    // Try to find available block
    if (available_blocks_.empty() && blocks_.size() < max_blocks_) {
        allocate_new_block();
    }
    
    if (!available_blocks_.empty()) {
        size_t block_idx = available_blocks_.front();
        available_blocks_.pop();
        blocks_[block_idx].in_use = true;
        return blocks_[block_idx].ptr;
    }
    
    // Pool exhausted, allocate directly
    return GPUMemoryAllocator::allocate(bytes);
}

void GPUMemoryPool::deallocate(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Check if this pointer belongs to our pool
    for (size_t i = 0; i < blocks_.size(); ++i) {
        if (blocks_[i].ptr == ptr && blocks_[i].in_use) {
            blocks_[i].in_use = false;
            available_blocks_.push(i);
            return;
        }
    }
    
    // Not from pool, deallocate directly
    GPUMemoryAllocator::deallocate(ptr);
}

void GPUMemoryPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& block : blocks_) {
        GPUMemoryAllocator::deallocate(block.ptr);
    }
    
    blocks_.clear();
    while (!available_blocks_.empty()) {
        available_blocks_.pop();
    }
    allocated_bytes_ = 0;
}

void GPUMemoryPool::allocate_new_block() {
    MemoryBlock block;
    block.ptr = GPUMemoryAllocator::allocate(block_size_);
    block.size = block_size_;
    block.in_use = false;
    
    blocks_.push_back(block);
    available_blocks_.push(blocks_.size() - 1);
    allocated_bytes_ += block_size_;
}

// GPUMemoryStats implementation
GPUMemoryStats::Stats GPUMemoryStats::stats_ = {0, 0, 0, 0, 0, 0};
std::mutex GPUMemoryStats::stats_mutex_;

void GPUMemoryStats::record_allocation(size_t bytes) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.total_allocations++;
    stats_.current_usage += bytes;
    stats_.total_bytes_allocated += bytes;
    stats_.peak_usage = std::max(stats_.peak_usage, stats_.current_usage);
}

void GPUMemoryStats::record_deallocation(size_t bytes) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.total_deallocations++;
    if (bytes > 0) {
        stats_.current_usage = (stats_.current_usage > bytes) ? stats_.current_usage - bytes : 0;
        stats_.total_bytes_deallocated += bytes;
    }
}

GPUMemoryStats::Stats GPUMemoryStats::get_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void GPUMemoryStats::reset_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = {0, 0, 0, 0, 0, 0};
}

void GPUMemoryStats::print_stats() {
    auto stats = get_stats();
    
    INFO("GPU Memory Statistics:");
    INFO("  Allocations: " << stats.total_allocations);
    INFO("  Deallocations: " << stats.total_deallocations);
    INFO("  Active allocations: " << (stats.total_allocations - stats.total_deallocations));
    INFO("  Current usage: " << (stats.current_usage / (1024*1024)) << " MB");
    INFO("  Peak usage: " << (stats.peak_usage / (1024*1024)) << " MB");
    INFO("  Total allocated: " << (stats.total_bytes_allocated / (1024*1024)) << " MB");
    INFO("  Total deallocated: " << (stats.total_bytes_deallocated / (1024*1024)) << " MB");
    
    auto& device = GPUDevice::instance();
    if (device.initialize()) {
        size_t device_total = device.get_total_memory();
        double usage_percent = (static_cast<double>(stats.current_usage) / device_total) * 100.0;
        INFO("  Device usage: " << std::fixed << std::setprecision(1) << usage_percent << "%");
    }
}

} // namespace gpu

#endif // SPADES_GPU_SUPPORT 