//***************************************************************************
//* Copyright (c) 2023-2024 SPAdes team
//* Copyright (c) 2020-2022 Saint Petersburg State University
//* All Rights Reserved
//* See file LICENSE for details.
//***************************************************************************

#pragma once

#ifdef SPADES_GPU_SUPPORT

#include "utils/gpu/gpu_memory.hpp"
#include "utils/gpu/gpu_device.hpp"

#ifdef SPADES_CUDA_SUPPORT
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#endif

#include <cstdint>
#include <vector>
#include <memory>

namespace gpu {

// GPU hash table for k-mer counting
class GPUHashTable {
public:
    struct Entry {
        uint64_t key;
        uint32_t value;
        uint32_t status; // 0=empty, 1=inserting, 2=occupied
    };

    GPUHashTable(size_t initial_capacity = 1024 * 1024);
    ~GPUHashTable();

    // Initialize the hash table
    bool initialize();
    void clear();
    
    // Resize operations
    void resize(size_t new_capacity);
    void rehash();
    
    // Insert k-mers (batch operation)
    void insert_kmers(const uint64_t* kmers, size_t count, cudaStream_t stream = 0);
    
    // Count k-mers (batch operation)
    void count_kmers(const uint64_t* kmers, size_t count, cudaStream_t stream = 0);
    
    // Retrieve results
    size_t extract_results(std::vector<std::pair<uint64_t, uint32_t>>& results);
    
    // Statistics
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    double load_factor() const { return capacity_ > 0 ? static_cast<double>(size_) / capacity_ : 0.0; }
    
    // Memory usage
    size_t memory_usage() const;
    void print_stats() const;

private:
    size_t capacity_;
    size_t size_;
    
    // GPU memory
    Entry* d_table_;
    uint32_t* d_size_counter_;
    
    // Host memory for results
    std::vector<Entry> h_results_;
    
    // Configuration
    static constexpr double MAX_LOAD_FACTOR = 0.7;
    static constexpr uint32_t EMPTY_STATUS = 0;
    static constexpr uint32_t INSERTING_STATUS = 1;
    static constexpr uint32_t OCCUPIED_STATUS = 2;
    
    void allocate_gpu_memory();
    void deallocate_gpu_memory();
    void initialize_table();
    
    // Disable copy operations
    GPUHashTable(const GPUHashTable&) = delete;
    GPUHashTable& operator=(const GPUHashTable&) = delete;
};

// Alternative: Cuckoo hash table for better performance
class GPUCuckooHashTable {
public:
    struct Bucket {
        static constexpr uint32_t BUCKET_SIZE = 4;
        uint64_t keys[BUCKET_SIZE];
        uint32_t values[BUCKET_SIZE];
        uint32_t count;
    };

    GPUCuckooHashTable(size_t num_buckets = 256 * 1024);
    ~GPUCuckooHashTable();

    bool initialize();
    void clear();
    
    // K-mer operations
    void insert_kmers(const uint64_t* kmers, size_t count, cudaStream_t stream = 0);
    size_t extract_results(std::vector<std::pair<uint64_t, uint32_t>>& results);
    
    // Statistics
    size_t capacity() const { return num_buckets_ * Bucket::BUCKET_SIZE; }
    size_t memory_usage() const;

private:
    size_t num_buckets_;
    Bucket* d_buckets_;
    uint32_t* d_overflow_counter_;
    
    void allocate_gpu_memory();
    void deallocate_gpu_memory();
    void initialize_buckets();
};

// Sorted k-mer counter using GPU sort primitives
class GPUSortedKMerCounter {
public:
    GPUSortedKMerCounter();
    ~GPUSortedKMerCounter();

    bool initialize();
    
    // Count k-mers by sorting and grouping
    void count_kmers(const uint64_t* kmers, size_t count, 
                    std::vector<std::pair<uint64_t, uint32_t>>& results,
                    cudaStream_t stream = 0);
    
    // Batch processing for large datasets
    void count_kmers_batched(const uint64_t* kmers, size_t count,
                           std::vector<std::pair<uint64_t, uint32_t>>& results,
                           size_t batch_size = 10 * 1024 * 1024,
                           cudaStream_t stream = 0);

private:
    // GPU memory for sorting
    uint64_t* d_keys_;
    uint64_t* d_keys_out_;
    uint32_t* d_values_;
    uint32_t* d_values_out_;
    uint32_t* d_unique_keys_count_;
    
    // Temporary storage for CUB operations
    void* d_temp_storage_;
    size_t temp_storage_bytes_;
    
    size_t allocated_capacity_;
    
    void allocate_memory(size_t capacity);
    void deallocate_memory();
    void ensure_capacity(size_t required_capacity);
};

// High-level k-mer counter interface
class GPUKMerCounter {
public:
    enum class Strategy {
        HASH_TABLE,
        CUCKOO_HASH,
        SORTED
    };

    GPUKMerCounter(Strategy strategy = Strategy::SORTED);
    ~GPUKMerCounter();

    bool initialize();
    
    // Count k-mers from sequences
    void count_kmers_from_sequences(
        const char* sequences,
        const uint32_t* seq_offsets,
        const uint32_t* seq_lengths,
        uint32_t num_sequences,
        uint32_t k,
        std::vector<std::pair<uint64_t, uint32_t>>& results,
        cudaStream_t stream = 0
    );
    
    // Direct k-mer counting
    void count_kmers(const uint64_t* kmers, size_t count,
                    std::vector<std::pair<uint64_t, uint32_t>>& results,
                    cudaStream_t stream = 0);
    
    // Performance statistics
    void print_performance_stats() const;
    void reset_stats();

private:
    Strategy strategy_;
    std::unique_ptr<GPUHashTable> hash_table_;
    std::unique_ptr<GPUCuckooHashTable> cuckoo_table_;
    std::unique_ptr<GPUSortedKMerCounter> sorted_counter_;
    
    // Performance tracking
    mutable size_t total_kmers_processed_;
    mutable double total_processing_time_;
    mutable size_t num_operations_;
    
    void update_stats(size_t kmers_count, double time) const;
};

} // namespace gpu

#else // SPADES_GPU_SUPPORT not defined

namespace gpu {

// Dummy implementations when GPU support is disabled
class GPUHashTable {
public:
    GPUHashTable(size_t = 1024 * 1024) {}
    bool initialize() { return false; }
    void clear() {}
    void insert_kmers(const uint64_t*, size_t, cudaStream_t = 0) {}
    size_t extract_results(std::vector<std::pair<uint64_t, uint32_t>>&) { return 0; }
    size_t size() const { return 0; }
    size_t capacity() const { return 0; }
    double load_factor() const { return 0.0; }
    size_t memory_usage() const { return 0; }
    void print_stats() const {}
};

class GPUKMerCounter {
public:
    enum class Strategy { HASH_TABLE, CUCKOO_HASH, SORTED };
    GPUKMerCounter(Strategy = Strategy::SORTED) {}
    bool initialize() { return false; }
    void count_kmers(const uint64_t*, size_t, std::vector<std::pair<uint64_t, uint32_t>>&, cudaStream_t = 0) {}
    void print_performance_stats() const {}
    void reset_stats() {}
};

} // namespace gpu

#endif // SPADES_GPU_SUPPORT 