//***************************************************************************
//* Copyright (c) 2023-2024 SPAdes team
//* Copyright (c) 2020-2022 Saint Petersburg State University
//* All Rights Reserved
//* See file LICENSE for details.
//***************************************************************************

#ifdef SPADES_CUDA_SUPPORT

#include "gpu_hash_table.hpp"
#include "gpu_kmer_kernels.cuh"
#include "utils/logger/logger.hpp"

#include <chrono>
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

namespace gpu {

// Hash table kernels
__global__ void hash_table_insert_kernel(
    GPUHashTable::Entry* __restrict__ table,
    const uint64_t* __restrict__ kmers,
    size_t count,
    size_t capacity,
    uint32_t* __restrict__ size_counter
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    
    for (uint32_t i = tid; i < count; i += stride) {
        uint64_t kmer = kmers[i];
        uint32_t hash = kernels::hash_function(kmer) % capacity;
        
        // Linear probing
        while (true) {
            uint32_t old_status = atomicCAS(&table[hash].status, 
                                          GPUHashTable::EMPTY_STATUS, 
                                          GPUHashTable::INSERTING_STATUS);
            
            if (old_status == GPUHashTable::EMPTY_STATUS) {
                // Successfully claimed empty slot
                table[hash].key = kmer;
                table[hash].value = 1;
                __threadfence();
                table[hash].status = GPUHashTable::OCCUPIED_STATUS;
                atomicAdd(size_counter, 1);
                break;
            } else if (old_status == GPUHashTable::OCCUPIED_STATUS && table[hash].key == kmer) {
                // K-mer already exists, increment count
                atomicAdd(&table[hash].value, 1);
                break;
            } else {
                // Collision or slot being inserted, try next slot
                hash = (hash + 1) % capacity;
            }
        }
    }
}

__global__ void hash_table_clear_kernel(GPUHashTable::Entry* table, size_t capacity) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    
    for (uint32_t i = tid; i < capacity; i += stride) {
        table[i].key = 0;
        table[i].value = 0;
        table[i].status = GPUHashTable::EMPTY_STATUS;
    }
}

__global__ void compact_results_kernel(
    const GPUHashTable::Entry* __restrict__ table,
    GPUHashTable::Entry* __restrict__ results,
    size_t capacity,
    uint32_t* __restrict__ result_count
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    
    for (uint32_t i = tid; i < capacity; i += stride) {
        if (table[i].status == GPUHashTable::OCCUPIED_STATUS) {
            uint32_t pos = atomicAdd(result_count, 1);
            results[pos] = table[i];
        }
    }
}

// GPUHashTable implementation
GPUHashTable::GPUHashTable(size_t initial_capacity) 
    : capacity_(initial_capacity), size_(0), d_table_(nullptr), d_size_counter_(nullptr) {
}

GPUHashTable::~GPUHashTable() {
    deallocate_gpu_memory();
}

bool GPUHashTable::initialize() {
    try {
        allocate_gpu_memory();
        initialize_table();
        return true;
    } catch (const std::exception& e) {
        INFO("Failed to initialize GPU hash table: " << e.what());
        return false;
    }
}

void GPUHashTable::clear() {
    if (!d_table_) return;
    
    dim3 block(256);
    dim3 grid((capacity_ + block.x - 1) / block.x);
    
    hash_table_clear_kernel<<<grid, block>>>(d_table_, capacity_);
    GPU_CHECK(cudaDeviceSynchronize());
    
    uint32_t zero = 0;
    GPU_CHECK(cudaMemcpy(d_size_counter_, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
    size_ = 0;
}

void GPUHashTable::insert_kmers(const uint64_t* kmers, size_t count, cudaStream_t stream) {
    if (!d_table_ || count == 0) return;
    
    // Check if resize is needed
    if (static_cast<double>(size_ + count) / capacity_ > MAX_LOAD_FACTOR) {
        size_t new_capacity = capacity_ * 2;
        while (static_cast<double>(size_ + count) / new_capacity > MAX_LOAD_FACTOR) {
            new_capacity *= 2;
        }
        resize(new_capacity);
    }
    
    dim3 block(256);
    dim3 grid(std::min((count + block.x - 1) / block.x, (size_t)2048));
    
    hash_table_insert_kernel<<<grid, block, 0, stream>>>(
        d_table_, kmers, count, capacity_, d_size_counter_
    );
    
    // Update size (this requires synchronization)
    if (stream == 0) {
        GPU_CHECK(cudaDeviceSynchronize());
        uint32_t new_size;
        GPU_CHECK(cudaMemcpy(&new_size, d_size_counter_, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        size_ = new_size;
    }
}

void GPUHashTable::count_kmers(const uint64_t* kmers, size_t count, cudaStream_t stream) {
    insert_kmers(kmers, count, stream);
}

size_t GPUHashTable::extract_results(std::vector<std::pair<uint64_t, uint32_t>>& results) {
    if (!d_table_ || size_ == 0) {
        results.clear();
        return 0;
    }
    
    // Allocate device memory for compacted results
    Entry* d_results;
    uint32_t* d_result_count;
    
    GPU_CHECK(cudaMalloc(&d_results, size_ * sizeof(Entry)));
    GPU_CHECK(cudaMalloc(&d_result_count, sizeof(uint32_t)));
    
    uint32_t zero = 0;
    GPU_CHECK(cudaMemcpy(d_result_count, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    dim3 block(256);
    dim3 grid((capacity_ + block.x - 1) / block.x);
    
    compact_results_kernel<<<grid, block>>>(d_table_, d_results, capacity_, d_result_count);
    GPU_CHECK(cudaDeviceSynchronize());
    
    // Get actual result count
    uint32_t actual_count;
    GPU_CHECK(cudaMemcpy(&actual_count, d_result_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    // Copy results to host
    h_results_.resize(actual_count);
    GPU_CHECK(cudaMemcpy(h_results_.data(), d_results, actual_count * sizeof(Entry), cudaMemcpyDeviceToHost));
    
    // Convert to output format
    results.clear();
    results.reserve(actual_count);
    for (const auto& entry : h_results_) {
        results.emplace_back(entry.key, entry.value);
    }
    
    // Cleanup
    GPU_CHECK(cudaFree(d_results));
    GPU_CHECK(cudaFree(d_result_count));
    
    return actual_count;
}

void GPUHashTable::resize(size_t new_capacity) {
    if (new_capacity <= capacity_) return;
    
    // Save old data
    std::vector<std::pair<uint64_t, uint32_t>> old_data;
    extract_results(old_data);
    
    // Reallocate
    deallocate_gpu_memory();
    capacity_ = new_capacity;
    allocate_gpu_memory();
    initialize_table();
    
    // Reinsert data
    if (!old_data.empty()) {
        std::vector<uint64_t> keys;
        keys.reserve(old_data.size());
        for (const auto& pair : old_data) {
            for (uint32_t i = 0; i < pair.second; ++i) {
                keys.push_back(pair.first);
            }
        }
        
        uint64_t* d_keys;
        GPU_CHECK(cudaMalloc(&d_keys, keys.size() * sizeof(uint64_t)));
        GPU_CHECK(cudaMemcpy(d_keys, keys.data(), keys.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
        
        insert_kmers(d_keys, keys.size());
        
        GPU_CHECK(cudaFree(d_keys));
    }
}

size_t GPUHashTable::memory_usage() const {
    return capacity_ * sizeof(Entry) + sizeof(uint32_t);
}

void GPUHashTable::print_stats() const {
    INFO("GPU Hash Table Statistics:");
    INFO("  Capacity: " << capacity_);
    INFO("  Size: " << size_);
    INFO("  Load factor: " << std::fixed << std::setprecision(3) << load_factor());
    INFO("  Memory usage: " << (memory_usage() / (1024 * 1024)) << " MB");
}

void GPUHashTable::allocate_gpu_memory() {
    GPU_CHECK(cudaMalloc(&d_table_, capacity_ * sizeof(Entry)));
    GPU_CHECK(cudaMalloc(&d_size_counter_, sizeof(uint32_t)));
}

void GPUHashTable::deallocate_gpu_memory() {
    if (d_table_) {
        GPU_CHECK(cudaFree(d_table_));
        d_table_ = nullptr;
    }
    if (d_size_counter_) {
        GPU_CHECK(cudaFree(d_size_counter_));
        d_size_counter_ = nullptr;
    }
}

void GPUHashTable::initialize_table() {
    clear();
}

// GPUSortedKMerCounter implementation
GPUSortedKMerCounter::GPUSortedKMerCounter() 
    : d_keys_(nullptr), d_keys_out_(nullptr), d_values_(nullptr), d_values_out_(nullptr),
      d_unique_keys_count_(nullptr), d_temp_storage_(nullptr), temp_storage_bytes_(0),
      allocated_capacity_(0) {
}

GPUSortedKMerCounter::~GPUSortedKMerCounter() {
    deallocate_memory();
}

bool GPUSortedKMerCounter::initialize() {
    try {
        // Initialize with a reasonable capacity
        allocate_memory(1024 * 1024);
        return true;
    } catch (const std::exception& e) {
        INFO("Failed to initialize GPU sorted k-mer counter: " << e.what());
        return false;
    }
}

void GPUSortedKMerCounter::count_kmers(const uint64_t* kmers, size_t count,
                                      std::vector<std::pair<uint64_t, uint32_t>>& results,
                                      cudaStream_t stream) {
    if (count == 0) {
        results.clear();
        return;
    }
    
    ensure_capacity(count);
    
    // Copy k-mers to device
    GPU_CHECK(cudaMemcpyAsync(d_keys_, kmers, count * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    
    // Initialize values to 1
    thrust::device_ptr<uint32_t> values_ptr(d_values_);
    thrust::fill(thrust::cuda::par.on(stream), values_ptr, values_ptr + count, 1);
    
    // Sort keys and values together
    thrust::device_ptr<uint64_t> keys_ptr(d_keys_);
    thrust::sort_by_key(thrust::cuda::par.on(stream), keys_ptr, keys_ptr + count, values_ptr);
    
    // Reduce by key to count occurrences
    thrust::device_ptr<uint64_t> keys_out_ptr(d_keys_out_);
    thrust::device_ptr<uint32_t> values_out_ptr(d_values_out_);
    
    auto end_pair = thrust::reduce_by_key(
        thrust::cuda::par.on(stream),
        keys_ptr, keys_ptr + count,
        values_ptr,
        keys_out_ptr,
        values_out_ptr
    );
    
    size_t unique_count = end_pair.first - keys_out_ptr;
    
    // Copy results back to host
    results.resize(unique_count);
    
    std::vector<uint64_t> host_keys(unique_count);
    std::vector<uint32_t> host_values(unique_count);
    
    GPU_CHECK(cudaMemcpyAsync(host_keys.data(), d_keys_out_, unique_count * sizeof(uint64_t), 
                             cudaMemcpyDeviceToHost, stream));
    GPU_CHECK(cudaMemcpyAsync(host_values.data(), d_values_out_, unique_count * sizeof(uint32_t), 
                             cudaMemcpyDeviceToHost, stream));
    
    GPU_CHECK(cudaStreamSynchronize(stream));
    
    for (size_t i = 0; i < unique_count; ++i) {
        results[i] = std::make_pair(host_keys[i], host_values[i]);
    }
}

void GPUSortedKMerCounter::count_kmers_batched(const uint64_t* kmers, size_t count,
                                              std::vector<std::pair<uint64_t, uint32_t>>& results,
                                              size_t batch_size, cudaStream_t stream) {
    results.clear();
    
    for (size_t offset = 0; offset < count; offset += batch_size) {
        size_t current_batch_size = std::min(batch_size, count - offset);
        
        std::vector<std::pair<uint64_t, uint32_t>> batch_results;
        count_kmers(kmers + offset, current_batch_size, batch_results, stream);
        
        // Merge with existing results
        if (results.empty()) {
            results = std::move(batch_results);
        } else {
            // Merge sorted results
            std::vector<std::pair<uint64_t, uint32_t>> merged;
            merged.reserve(results.size() + batch_results.size());
            
            std::merge(results.begin(), results.end(),
                      batch_results.begin(), batch_results.end(),
                      std::back_inserter(merged));
            
            // Combine counts for duplicate k-mers
            results.clear();
            if (!merged.empty()) {
                results.push_back(merged[0]);
                for (size_t i = 1; i < merged.size(); ++i) {
                    if (merged[i].first == results.back().first) {
                        results.back().second += merged[i].second;
                    } else {
                        results.push_back(merged[i]);
                    }
                }
            }
        }
    }
}

void GPUSortedKMerCounter::allocate_memory(size_t capacity) {
    deallocate_memory();
    
    GPU_CHECK(cudaMalloc(&d_keys_, capacity * sizeof(uint64_t)));
    GPU_CHECK(cudaMalloc(&d_keys_out_, capacity * sizeof(uint64_t)));
    GPU_CHECK(cudaMalloc(&d_values_, capacity * sizeof(uint32_t)));
    GPU_CHECK(cudaMalloc(&d_values_out_, capacity * sizeof(uint32_t)));
    GPU_CHECK(cudaMalloc(&d_unique_keys_count_, sizeof(uint32_t)));
    
    allocated_capacity_ = capacity;
}

void GPUSortedKMerCounter::deallocate_memory() {
    if (d_keys_) { GPU_CHECK(cudaFree(d_keys_)); d_keys_ = nullptr; }
    if (d_keys_out_) { GPU_CHECK(cudaFree(d_keys_out_)); d_keys_out_ = nullptr; }
    if (d_values_) { GPU_CHECK(cudaFree(d_values_)); d_values_ = nullptr; }
    if (d_values_out_) { GPU_CHECK(cudaFree(d_values_out_)); d_values_out_ = nullptr; }
    if (d_unique_keys_count_) { GPU_CHECK(cudaFree(d_unique_keys_count_)); d_unique_keys_count_ = nullptr; }
    if (d_temp_storage_) { GPU_CHECK(cudaFree(d_temp_storage_)); d_temp_storage_ = nullptr; }
    
    allocated_capacity_ = 0;
    temp_storage_bytes_ = 0;
}

void GPUSortedKMerCounter::ensure_capacity(size_t required_capacity) {
    if (required_capacity > allocated_capacity_) {
        size_t new_capacity = allocated_capacity_;
        while (new_capacity < required_capacity) {
            new_capacity = std::max(new_capacity * 2, required_capacity);
        }
        allocate_memory(new_capacity);
    }
}

// GPUKMerCounter implementation
GPUKMerCounter::GPUKMerCounter(Strategy strategy) 
    : strategy_(strategy), total_kmers_processed_(0), total_processing_time_(0.0), num_operations_(0) {
    
    switch (strategy_) {
        case Strategy::HASH_TABLE:
            hash_table_ = std::make_unique<GPUHashTable>();
            break;
        case Strategy::CUCKOO_HASH:
            cuckoo_table_ = std::make_unique<GPUCuckooHashTable>();
            break;
        case Strategy::SORTED:
            sorted_counter_ = std::make_unique<GPUSortedKMerCounter>();
            break;
    }
}

GPUKMerCounter::~GPUKMerCounter() = default;

bool GPUKMerCounter::initialize() {
    switch (strategy_) {
        case Strategy::HASH_TABLE:
            return hash_table_->initialize();
        case Strategy::CUCKOO_HASH:
            return cuckoo_table_->initialize();
        case Strategy::SORTED:
            return sorted_counter_->initialize();
    }
    return false;
}

void GPUKMerCounter::count_kmers(const uint64_t* kmers, size_t count,
                                std::vector<std::pair<uint64_t, uint32_t>>& results,
                                cudaStream_t stream) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    switch (strategy_) {
        case Strategy::HASH_TABLE:
            hash_table_->count_kmers(kmers, count, stream);
            GPU_CHECK(cudaStreamSynchronize(stream));
            hash_table_->extract_results(results);
            break;
        case Strategy::SORTED:
            sorted_counter_->count_kmers(kmers, count, results, stream);
            break;
        case Strategy::CUCKOO_HASH:
            cuckoo_table_->insert_kmers(kmers, count, stream);
            GPU_CHECK(cudaStreamSynchronize(stream));
            cuckoo_table_->extract_results(results);
            break;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
    
    update_stats(count, elapsed_time);
}

void GPUKMerCounter::count_kmers_from_sequences(
    const char* sequences,
    const uint32_t* seq_offsets,
    const uint32_t* seq_lengths,
    uint32_t num_sequences,
    uint32_t k,
    std::vector<std::pair<uint64_t, uint32_t>>& results,
    cudaStream_t stream
) {
    // Estimate maximum k-mers
    size_t max_kmers = 0;
    for (uint32_t i = 0; i < num_sequences; ++i) {
        if (seq_lengths[i] >= k) {
            max_kmers += (seq_lengths[i] - k + 1);
        }
    }
    
    if (max_kmers == 0) {
        results.clear();
        return;
    }
    
    // Allocate GPU memory for k-mers
    uint64_t* d_kmers;
    uint32_t* d_kmer_counts;
    
    GPU_CHECK(cudaMalloc(&d_kmers, max_kmers * sizeof(uint64_t)));
    GPU_CHECK(cudaMalloc(&d_kmer_counts, num_sequences * sizeof(uint32_t)));
    
    // Extract k-mers using GPU kernel
    kernels::launch_extract_kmers_kernel(
        sequences, seq_offsets, seq_lengths, d_kmers, d_kmer_counts,
        num_sequences, k, max_kmers / num_sequences, stream
    );
    
    // Count actual number of k-mers extracted
    thrust::device_ptr<uint32_t> counts_ptr(d_kmer_counts);
    size_t total_kmers = thrust::reduce(thrust::cuda::par.on(stream), counts_ptr, counts_ptr + num_sequences);
    
    // Count k-mers
    count_kmers(d_kmers, total_kmers, results, stream);
    
    // Cleanup
    GPU_CHECK(cudaFree(d_kmers));
    GPU_CHECK(cudaFree(d_kmer_counts));
}

void GPUKMerCounter::print_performance_stats() const {
    if (num_operations_ == 0) {
        INFO("No GPU k-mer counting operations performed yet");
        return;
    }
    
    double avg_time = total_processing_time_ / num_operations_;
    double throughput = total_kmers_processed_ / total_processing_time_;
    
    INFO("GPU K-mer Counter Performance Statistics:");
    INFO("  Strategy: " << (strategy_ == Strategy::SORTED ? "Sorted" : 
                           strategy_ == Strategy::HASH_TABLE ? "Hash Table" : "Cuckoo Hash"));
    INFO("  Total operations: " << num_operations_);
    INFO("  Total k-mers processed: " << total_kmers_processed_);
    INFO("  Total processing time: " << std::fixed << std::setprecision(3) << total_processing_time_ << " seconds");
    INFO("  Average time per operation: " << std::fixed << std::setprecision(3) << avg_time << " seconds");
    INFO("  Throughput: " << std::fixed << std::setprecision(1) << (throughput / 1e6) << " million k-mers/second");
}

void GPUKMerCounter::reset_stats() {
    total_kmers_processed_ = 0;
    total_processing_time_ = 0.0;
    num_operations_ = 0;
}

void GPUKMerCounter::update_stats(size_t kmers_count, double time) const {
    total_kmers_processed_ += kmers_count;
    total_processing_time_ += time;
    num_operations_++;
}

} // namespace gpu

#endif // SPADES_CUDA_SUPPORT 