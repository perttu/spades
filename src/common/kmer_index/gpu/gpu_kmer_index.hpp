//***************************************************************************
//* Copyright (c) 2023-2024 SPAdes team
//* Copyright (c) 2020-2022 Saint Petersburg State University
//* All Rights Reserved
//* See file LICENSE for details.
//***************************************************************************

#pragma once

#include "gpu_hash_table.hpp"
#include "utils/gpu/gpu_device.hpp"
#include "sequence/sequence.hpp"
#include "io/reads/single_read.hpp"
#include "io/reads/paired_read.hpp"

#ifdef SPADES_GPU_SUPPORT
#include <cuda_runtime.h>
#endif

#include <vector>
#include <unordered_map>
#include <memory>

namespace gpu {

// Forward declarations for SPAdes types
template<class K> class KMerIndex;

// GPU-accelerated k-mer index for SPAdES integration
template<class K>
class GPUKMerIndex {
public:
    using KMerT = K;
    using CountT = uint32_t;
    using KMerCount = std::pair<KMerT, CountT>;

    GPUKMerIndex(unsigned k);
    ~GPUKMerIndex();

    // Initialize GPU support
    bool initialize();
    void shutdown();

    // Build index from reads
    void add_reads(const std::vector<io::SingleRead>& reads);
    void add_reads(const std::vector<io::PairedRead>& reads);
    
    // Build index from sequences
    void add_sequences(const std::vector<Sequence>& sequences);
    void add_sequence(const Sequence& sequence);
    
    // K-mer operations
    CountT count(const KMerT& kmer) const;
    bool contains(const KMerT& kmer) const;
    
    // Bulk operations
    void count_batch(const std::vector<KMerT>& kmers, std::vector<CountT>& counts) const;
    
    // Index statistics
    size_t size() const { return kmer_counts_.size(); }
    size_t unique_kmers() const { return kmer_counts_.size(); }
    uint64_t total_kmers() const;
    
    // Memory usage
    size_t memory_usage() const;
    void print_stats() const;
    
    // Export to CPU k-mer index for compatibility
    void export_to_cpu_index(KMerIndex<K>& cpu_index) const;
    
    // Performance mode selection
    enum class PerformanceMode {
        BALANCED,    // Use GPU for large batches, CPU for small ones
        GPU_ONLY,    // Force GPU processing
        ADAPTIVE     // Dynamically choose based on workload
    };
    
    void set_performance_mode(PerformanceMode mode) { performance_mode_ = mode; }
    PerformanceMode get_performance_mode() const { return performance_mode_; }

private:
    unsigned k_;
    bool initialized_;
    PerformanceMode performance_mode_;
    
    // GPU components
    std::unique_ptr<GPUKMerCounter> gpu_counter_;
    
    // CPU fallback and result storage
    std::unordered_map<KMerT, CountT> kmer_counts_;
    
    // Performance thresholds
    static constexpr size_t GPU_BATCH_THRESHOLD = 10000;  // Minimum batch size for GPU
    static constexpr size_t GPU_MEMORY_THRESHOLD = 100 * 1024 * 1024;  // 100MB min GPU memory
    
    // Internal methods
    void process_sequences_gpu(const std::vector<Sequence>& sequences);
    void process_sequences_cpu(const std::vector<Sequence>& sequences);
    bool should_use_gpu(size_t sequence_count, size_t total_length) const;
    void merge_gpu_results(const std::vector<std::pair<uint64_t, uint32_t>>& gpu_results);
    
    // Sequence conversion for GPU
    void prepare_sequences_for_gpu(const std::vector<Sequence>& sequences,
                                   std::vector<char>& gpu_sequences,
                                   std::vector<uint32_t>& offsets,
                                   std::vector<uint32_t>& lengths) const;
};

// Factory function for creating GPU k-mer indices
template<class K>
std::unique_ptr<GPUKMerIndex<K>> create_gpu_kmer_index(unsigned k) {
    auto index = std::make_unique<GPUKMerIndex<K>>(k);
    if (index->initialize()) {
        return index;
    }
    return nullptr;
}

// Utility function to check GPU k-mer counting capability
bool is_gpu_kmer_counting_available();

// Benchmark GPU vs CPU performance for given workload
struct PerformanceBenchmark {
    double gpu_time;
    double cpu_time;
    double gpu_throughput;  // k-mers per second
    double cpu_throughput;
    size_t gpu_memory_used;
    bool gpu_successful;
};

PerformanceBenchmark benchmark_kmer_counting(unsigned k, 
                                           const std::vector<Sequence>& test_sequences);

// Configuration for GPU k-mer index
struct GPUKMerIndexConfig {
    GPUKMerCounter::Strategy strategy = GPUKMerCounter::Strategy::SORTED;
    size_t batch_size = 1000000;  // k-mers per batch
    bool enable_memory_mapping = true;
    bool use_pinned_memory = true;
    double gpu_memory_fraction = 0.8;  // Fraction of GPU memory to use
    size_t min_gpu_batch_size = 10000;
    
    // Load from environment variables or config file
    static GPUKMerIndexConfig load_from_config();
    void print_config() const;
};

} // namespace gpu

// Integration with existing SPAdes types
namespace debruijn_graph {

// Specialized GPU k-mer index for DeBruijn graph construction
template<class Graph>
class GPUDeBruijnIndex {
public:
    using EdgeId = typename Graph::EdgeId;
    using VertexId = typename Graph::VertexId;
    using KMerT = typename Graph::KMerT;
    
    GPUDeBruijnIndex(const Graph& graph, unsigned k);
    ~GPUDeBruijnIndex();
    
    // Build index from graph edges
    bool build_from_graph();
    
    // Query operations
    EdgeId find_edge(const KMerT& kmer) const;
    std::vector<EdgeId> find_edges(const std::vector<KMerT>& kmers) const;
    
    // Statistics
    size_t size() const;
    void print_stats() const;

private:
    const Graph& graph_;
    unsigned k_;
    std::unique_ptr<gpu::GPUKMerIndex<KMerT>> gpu_index_;
    std::unordered_map<KMerT, EdgeId> kmer_to_edge_;
};

} // namespace debruijn_graph

#ifndef SPADES_GPU_SUPPORT

// Dummy implementations when GPU support is disabled
namespace gpu {

template<class K>
class GPUKMerIndex {
public:
    using KMerT = K;
    using CountT = uint32_t;
    
    GPUKMerIndex(unsigned) {}
    bool initialize() { return false; }
    void shutdown() {}
    void add_sequences(const std::vector<Sequence>&) {}
    void add_sequence(const Sequence&) {}
    CountT count(const KMerT&) const { return 0; }
    bool contains(const KMerT&) const { return false; }
    size_t size() const { return 0; }
    size_t memory_usage() const { return 0; }
    void print_stats() const {}
    
    enum class PerformanceMode { BALANCED, GPU_ONLY, ADAPTIVE };
    void set_performance_mode(PerformanceMode) {}
    PerformanceMode get_performance_mode() const { return PerformanceMode::BALANCED; }
};

template<class K>
std::unique_ptr<GPUKMerIndex<K>> create_gpu_kmer_index(unsigned) {
    return nullptr;
}

inline bool is_gpu_kmer_counting_available() { return false; }

struct PerformanceBenchmark {
    double gpu_time = 0;
    double cpu_time = 0;
    double gpu_throughput = 0;
    double cpu_throughput = 0;
    size_t gpu_memory_used = 0;
    bool gpu_successful = false;
};

inline PerformanceBenchmark benchmark_kmer_counting(unsigned, const std::vector<Sequence>&) {
    return PerformanceBenchmark{};
}

struct GPUKMerIndexConfig {
    static GPUKMerIndexConfig load_from_config() { return GPUKMerIndexConfig{}; }
    void print_config() const {}
};

} // namespace gpu

namespace debruijn_graph {

template<class Graph>
class GPUDeBruijnIndex {
public:
    using EdgeId = typename Graph::EdgeId;
    using KMerT = typename Graph::KMerT;
    
    GPUDeBruijnIndex(const Graph&, unsigned) {}
    bool build_from_graph() { return false; }
    EdgeId find_edge(const KMerT&) const { return EdgeId(); }
    std::vector<EdgeId> find_edges(const std::vector<KMerT>&) const { return {}; }
    size_t size() const { return 0; }
    void print_stats() const {}
};

} // namespace debruijn_graph

#endif // SPADES_GPU_SUPPORT 