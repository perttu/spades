//***************************************************************************
//* Copyright (c) 2023-2024 SPAdes team
//* Copyright (c) 2020-2022 Saint Petersburg State University
//* All Rights Reserved
//* See file LICENSE for details.
//***************************************************************************

#include "gpu_kmer_index.hpp"
#include "utils/logger/logger.hpp"
#include "sequence/nucl.hpp"

#ifdef SPADES_GPU_SUPPORT
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <thread>
#endif

namespace gpu {

#ifdef SPADES_GPU_SUPPORT

template<class K>
GPUKMerIndex<K>::GPUKMerIndex(unsigned k) 
    : k_(k), initialized_(false), performance_mode_(PerformanceMode::BALANCED) {
    gpu_counter_ = std::make_unique<GPUKMerCounter>(GPUKMerCounter::Strategy::SORTED);
}

template<class K>
GPUKMerIndex<K>::~GPUKMerIndex() {
    shutdown();
}

template<class K>
bool GPUKMerIndex<K>::initialize() {
    if (initialized_) {
        return true;
    }
    
    // Check if GPU is available
    auto& device = GPUDevice::instance();
    if (!device.initialize()) {
        INFO("GPU not available, falling back to CPU-only k-mer counting");
        return false;
    }
    
    // Check GPU memory requirements
    size_t available_memory = device.get_available_memory();
    if (available_memory < GPU_MEMORY_THRESHOLD) {
        INFO("Insufficient GPU memory (" << (available_memory / (1024*1024)) 
             << " MB), falling back to CPU-only k-mer counting");
        return false;
    }
    
    // Initialize GPU counter
    if (!gpu_counter_->initialize()) {
        INFO("Failed to initialize GPU k-mer counter");
        return false;
    }
    
    initialized_ = true;
    INFO("GPU k-mer index initialized successfully");
    INFO("  Available GPU memory: " << (available_memory / (1024*1024)) << " MB");
    INFO("  K-mer size: " << k_);
    
    return true;
}

template<class K>
void GPUKMerIndex<K>::shutdown() {
    if (initialized_) {
        gpu_counter_.reset();
        initialized_ = false;
        INFO("GPU k-mer index shutdown");
    }
}

template<class K>
void GPUKMerIndex<K>::add_reads(const std::vector<io::SingleRead>& reads) {
    std::vector<Sequence> sequences;
    sequences.reserve(reads.size());
    
    for (const auto& read : reads) {
        sequences.push_back(read.sequence());
    }
    
    add_sequences(sequences);
}

template<class K>
void GPUKMerIndex<K>::add_reads(const std::vector<io::PairedRead>& reads) {
    std::vector<Sequence> sequences;
    sequences.reserve(reads.size() * 2);
    
    for (const auto& read : reads) {
        sequences.push_back(read.first().sequence());
        sequences.push_back(read.second().sequence());
    }
    
    add_sequences(sequences);
}

template<class K>
void GPUKMerIndex<K>::add_sequences(const std::vector<Sequence>& sequences) {
    if (sequences.empty()) return;
    
    size_t total_length = 0;
    for (const auto& seq : sequences) {
        total_length += seq.size();
    }
    
    if (should_use_gpu(sequences.size(), total_length)) {
        process_sequences_gpu(sequences);
    } else {
        process_sequences_cpu(sequences);
    }
}

template<class K>
void GPUKMerIndex<K>::add_sequence(const Sequence& sequence) {
    std::vector<Sequence> sequences = {sequence};
    add_sequences(sequences);
}

template<class K>
typename GPUKMerIndex<K>::CountT GPUKMerIndex<K>::count(const KMerT& kmer) const {
    auto it = kmer_counts_.find(kmer);
    return (it != kmer_counts_.end()) ? it->second : 0;
}

template<class K>
bool GPUKMerIndex<K>::contains(const KMerT& kmer) const {
    return kmer_counts_.find(kmer) != kmer_counts_.end();
}

template<class K>
void GPUKMerIndex<K>::count_batch(const std::vector<KMerT>& kmers, std::vector<CountT>& counts) const {
    counts.clear();
    counts.reserve(kmers.size());
    
    for (const auto& kmer : kmers) {
        counts.push_back(count(kmer));
    }
}

template<class K>
uint64_t GPUKMerIndex<K>::total_kmers() const {
    uint64_t total = 0;
    for (const auto& pair : kmer_counts_) {
        total += pair.second;
    }
    return total;
}

template<class K>
size_t GPUKMerIndex<K>::memory_usage() const {
    size_t cpu_memory = kmer_counts_.size() * (sizeof(KMerT) + sizeof(CountT));
    
    size_t gpu_memory = 0;
    if (initialized_ && gpu_counter_) {
        // Estimate GPU memory usage
        auto& device = GPUDevice::instance();
        gpu_memory = device.get_total_memory() - device.get_available_memory();
    }
    
    return cpu_memory + gpu_memory;
}

template<class K>
void GPUKMerIndex<K>::print_stats() const {
    INFO("GPU K-mer Index Statistics:");
    INFO("  K-mer size: " << k_);
    INFO("  Unique k-mers: " << unique_kmers());
    INFO("  Total k-mers: " << total_kmers());
    INFO("  Memory usage: " << (memory_usage() / (1024*1024)) << " MB");
    INFO("  Performance mode: " << 
         (performance_mode_ == PerformanceMode::BALANCED ? "Balanced" :
          performance_mode_ == PerformanceMode::GPU_ONLY ? "GPU Only" : "Adaptive"));
    
    if (initialized_ && gpu_counter_) {
        gpu_counter_->print_performance_stats();
    }
}

template<class K>
void GPUKMerIndex<K>::process_sequences_gpu(const std::vector<Sequence>& sequences) {
    if (!initialized_) {
        INFO("GPU not initialized, falling back to CPU processing");
        process_sequences_cpu(sequences);
        return;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Prepare sequences for GPU
        std::vector<char> gpu_sequences;
        std::vector<uint32_t> offsets, lengths;
        prepare_sequences_for_gpu(sequences, gpu_sequences, offsets, lengths);
        
        // Allocate GPU memory and transfer data
        char* d_sequences;
        uint32_t* d_offsets;
        uint32_t* d_lengths;
        
        GPU_CHECK(cudaMalloc(&d_sequences, gpu_sequences.size()));
        GPU_CHECK(cudaMalloc(&d_offsets, offsets.size() * sizeof(uint32_t)));
        GPU_CHECK(cudaMalloc(&d_lengths, lengths.size() * sizeof(uint32_t)));
        
        GPU_CHECK(cudaMemcpy(d_sequences, gpu_sequences.data(), gpu_sequences.size(), cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(d_lengths, lengths.data(), lengths.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        
        // Process k-mers on GPU
        std::vector<std::pair<uint64_t, uint32_t>> gpu_results;
        gpu_counter_->count_kmers_from_sequences(
            d_sequences, d_offsets, d_lengths, 
            static_cast<uint32_t>(sequences.size()), k_, gpu_results
        );
        
        // Merge results into CPU index
        merge_gpu_results(gpu_results);
        
        // Cleanup GPU memory
        GPU_CHECK(cudaFree(d_sequences));
        GPU_CHECK(cudaFree(d_offsets));
        GPU_CHECK(cudaFree(d_lengths));
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end_time - start_time).count();
        
        INFO("GPU k-mer processing completed in " << std::fixed << std::setprecision(3) 
             << elapsed << " seconds");
        INFO("  Processed " << sequences.size() << " sequences");
        INFO("  Found " << gpu_results.size() << " unique k-mers");
        
    } catch (const std::exception& e) {
        INFO("GPU processing failed: " << e.what() << ", falling back to CPU");
        process_sequences_cpu(sequences);
    }
}

template<class K>
void GPUKMerIndex<K>::process_sequences_cpu(const std::vector<Sequence>& sequences) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    size_t processed_kmers = 0;
    
    for (const auto& sequence : sequences) {
        if (sequence.size() < k_) continue;
        
        for (size_t i = 0; i <= sequence.size() - k_; ++i) {
            KMerT kmer = sequence.substr(i, k_);
            
            // Skip k-mers with invalid nucleotides
            bool valid = true;
            for (size_t j = 0; j < k_; ++j) {
                if (!is_nucl(kmer[j])) {
                    valid = false;
                    break;
                }
            }
            
            if (valid) {
                // Use canonical k-mer (lexicographically smaller of kmer and reverse complement)
                KMerT rc_kmer = !kmer;
                KMerT canonical = (kmer < rc_kmer) ? kmer : rc_kmer;
                
                kmer_counts_[canonical]++;
                processed_kmers++;
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();
    
    INFO("CPU k-mer processing completed in " << std::fixed << std::setprecision(3) 
         << elapsed << " seconds");
    INFO("  Processed " << sequences.size() << " sequences");
    INFO("  Found " << processed_kmers << " k-mers (" << kmer_counts_.size() << " unique)");
}

template<class K>
bool GPUKMerIndex<K>::should_use_gpu(size_t sequence_count, size_t total_length) const {
    if (!initialized_) return false;
    
    switch (performance_mode_) {
        case PerformanceMode::GPU_ONLY:
            return true;
            
        case PerformanceMode::BALANCED: {
            // Estimate number of k-mers
            size_t estimated_kmers = (total_length >= k_) ? (total_length - k_ + 1) * sequence_count : 0;
            return estimated_kmers >= GPU_BATCH_THRESHOLD;
        }
        
        case PerformanceMode::ADAPTIVE: {
            // Use GPU if we have sufficient memory and workload
            auto& device = GPUDevice::instance();
            size_t available_memory = device.get_available_memory();
            size_t required_memory = total_length + sequence_count * sizeof(uint32_t) * 2;
            
            return (available_memory > required_memory * 2) && 
                   (sequence_count >= GPU_BATCH_THRESHOLD / 1000);
        }
    }
    
    return false;
}

template<class K>
void GPUKMerIndex<K>::merge_gpu_results(const std::vector<std::pair<uint64_t, uint32_t>>& gpu_results) {
    for (const auto& result : gpu_results) {
        // Convert uint64_t back to K-mer type
        KMerT kmer(k_, result.first);
        kmer_counts_[kmer] += result.second;
    }
}

template<class K>
void GPUKMerIndex<K>::prepare_sequences_for_gpu(const std::vector<Sequence>& sequences,
                                               std::vector<char>& gpu_sequences,
                                               std::vector<uint32_t>& offsets,
                                               std::vector<uint32_t>& lengths) const {
    // Calculate total size needed
    size_t total_size = 0;
    for (const auto& seq : sequences) {
        total_size += seq.size();
    }
    
    gpu_sequences.clear();
    gpu_sequences.reserve(total_size);
    offsets.clear();
    offsets.reserve(sequences.size());
    lengths.clear();
    lengths.reserve(sequences.size());
    
    uint32_t current_offset = 0;
    
    for (const auto& seq : sequences) {
        offsets.push_back(current_offset);
        lengths.push_back(static_cast<uint32_t>(seq.size()));
        
        // Copy sequence data
        for (size_t i = 0; i < seq.size(); ++i) {
            gpu_sequences.push_back(seq[i]);
        }
        
        current_offset += static_cast<uint32_t>(seq.size());
    }
}

// Utility functions
bool is_gpu_kmer_counting_available() {
    auto& device = GPUDevice::instance();
    return device.initialize() && device.get_available_memory() >= 100 * 1024 * 1024; // 100MB minimum
}

PerformanceBenchmark benchmark_kmer_counting(unsigned k, const std::vector<Sequence>& test_sequences) {
    PerformanceBenchmark result = {};
    
    if (test_sequences.empty()) {
        return result;
    }
    
    // Test GPU performance
    auto gpu_index = create_gpu_kmer_index<RtSeq>(k);
    if (gpu_index) {
        auto start = std::chrono::high_resolution_clock::now();
        gpu_index->add_sequences(test_sequences);
        auto end = std::chrono::high_resolution_clock::now();
        
        result.gpu_time = std::chrono::duration<double>(end - start).count();
        result.gpu_successful = true;
        result.gpu_memory_used = gpu_index->memory_usage();
        
        uint64_t total_kmers = gpu_index->total_kmers();
        result.gpu_throughput = total_kmers / result.gpu_time;
    }
    
    // Test CPU performance
    {
        std::unordered_map<RtSeq, uint32_t> cpu_counts;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (const auto& sequence : test_sequences) {
            if (sequence.size() < k) continue;
            
            for (size_t i = 0; i <= sequence.size() - k; ++i) {
                RtSeq kmer = sequence.substr(i, k);
                
                bool valid = true;
                for (size_t j = 0; j < k; ++j) {
                    if (!is_nucl(kmer[j])) {
                        valid = false;
                        break;
                    }
                }
                
                if (valid) {
                    RtSeq rc_kmer = !kmer;
                    RtSeq canonical = (kmer < rc_kmer) ? kmer : rc_kmer;
                    cpu_counts[canonical]++;
                }
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        result.cpu_time = std::chrono::duration<double>(end - start).count();
        
        uint64_t total_kmers = 0;
        for (const auto& pair : cpu_counts) {
            total_kmers += pair.second;
        }
        result.cpu_throughput = total_kmers / result.cpu_time;
    }
    
    return result;
}

GPUKMerIndexConfig GPUKMerIndexConfig::load_from_config() {
    GPUKMerIndexConfig config;
    
    // Load from environment variables
    if (const char* strategy = std::getenv("SPADES_GPU_STRATEGY")) {
        std::string str_strategy(strategy);
        if (str_strategy == "hash") {
            config.strategy = GPUKMerCounter::Strategy::HASH_TABLE;
        } else if (str_strategy == "cuckoo") {
            config.strategy = GPUKMerCounter::Strategy::CUCKOO_HASH;
        } else if (str_strategy == "sorted") {
            config.strategy = GPUKMerCounter::Strategy::SORTED;
        }
    }
    
    if (const char* batch_size = std::getenv("SPADES_GPU_BATCH_SIZE")) {
        config.batch_size = std::stoul(batch_size);
    }
    
    if (const char* memory_fraction = std::getenv("SPADES_GPU_MEMORY_FRACTION")) {
        config.gpu_memory_fraction = std::stod(memory_fraction);
    }
    
    return config;
}

void GPUKMerIndexConfig::print_config() const {
    INFO("GPU K-mer Index Configuration:");
    INFO("  Strategy: " << (strategy == GPUKMerCounter::Strategy::SORTED ? "Sorted" :
                           strategy == GPUKMerCounter::Strategy::HASH_TABLE ? "Hash Table" : "Cuckoo Hash"));
    INFO("  Batch size: " << batch_size);
    INFO("  Enable memory mapping: " << (enable_memory_mapping ? "Yes" : "No"));
    INFO("  Use pinned memory: " << (use_pinned_memory ? "Yes" : "No"));
    INFO("  GPU memory fraction: " << std::fixed << std::setprecision(2) << gpu_memory_fraction);
    INFO("  Min GPU batch size: " << min_gpu_batch_size);
}

// Explicit template instantiations for common k-mer types
template class GPUKMerIndex<RtSeq>;
template class GPUKMerIndex<Kmer<21>>;
template class GPUKMerIndex<Kmer<55>>;

#endif // SPADES_GPU_SUPPORT

} // namespace gpu 