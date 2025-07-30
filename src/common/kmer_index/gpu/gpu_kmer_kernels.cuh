//***************************************************************************
//* Copyright (c) 2023-2024 SPAdes team
//* Copyright (c) 2020-2022 Saint Petersburg State University
//* All Rights Reserved
//* See file LICENSE for details.
//***************************************************************************

#pragma once

#ifdef SPADES_CUDA_SUPPORT

#include <cuda_runtime.h>
#include <cstdint>

namespace gpu {
namespace kernels {

// Hash table status constants
constexpr uint32_t HASH_EMPTY = 0;
constexpr uint32_t HASH_INSERTING = 1;
constexpr uint32_t HASH_OCCUPIED = 2;

// Device function declarations
__device__ __forceinline__ uint32_t hash_function(uint64_t kmer);
__device__ __forceinline__ uint64_t reverse_complement(uint64_t kmer, uint32_t k);

// Kernel declarations
__global__ void init_nucleotide_map();

__global__ void extract_kmers_kernel(
    const char* __restrict__ sequences,
    const uint32_t* __restrict__ seq_offsets,
    const uint32_t* __restrict__ seq_lengths,
    uint64_t* __restrict__ kmers,
    uint32_t* __restrict__ kmer_counts,
    uint32_t num_sequences,
    uint32_t k,
    uint32_t max_kmers_per_seq
);

__global__ void count_kmers_kernel(
    const uint64_t* __restrict__ kmers,
    const uint32_t* __restrict__ kmer_counts,
    const uint32_t* __restrict__ kmer_offsets,
    uint64_t* __restrict__ hash_keys,
    uint32_t* __restrict__ hash_values,
    uint32_t* __restrict__ hash_status,
    uint32_t num_sequences,
    uint32_t hash_table_size
);

__global__ void extract_kmers_coalesced_kernel(
    const char* __restrict__ sequences,
    const uint32_t* __restrict__ seq_info,
    uint64_t* __restrict__ kmers,
    uint32_t* __restrict__ kmer_positions,
    uint32_t num_sequences,
    uint32_t k,
    uint32_t max_output_kmers
);

__global__ void sort_kmers_setup_kernel(
    const uint64_t* __restrict__ input_kmers,
    uint64_t* __restrict__ output_kmers,
    uint32_t* __restrict__ indices,
    uint32_t num_kmers
);

// Host interface functions
void initialize_gpu_nucleotide_map(cudaStream_t stream = 0);

void launch_extract_kmers_kernel(
    const char* sequences,
    const uint32_t* seq_offsets,
    const uint32_t* seq_lengths,
    uint64_t* kmers,
    uint32_t* kmer_counts,
    uint32_t num_sequences,
    uint32_t k,
    uint32_t max_kmers_per_seq,
    cudaStream_t stream = 0
);

void launch_count_kmers_kernel(
    const uint64_t* kmers,
    const uint32_t* kmer_counts,
    const uint32_t* kmer_offsets,
    uint64_t* hash_keys,
    uint32_t* hash_values,
    uint32_t* hash_status,
    uint32_t num_sequences,
    uint32_t hash_table_size,
    cudaStream_t stream = 0
);

void launch_extract_kmers_coalesced_kernel(
    const char* sequences,
    const uint32_t* seq_info,
    uint64_t* kmers,
    uint32_t* kmer_positions,
    uint32_t num_sequences,
    uint32_t k,
    uint32_t max_output_kmers,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace gpu

#endif // SPADES_CUDA_SUPPORT 