//***************************************************************************
//* Copyright (c) 2023-2024 SPAdes team
//* Copyright (c) 2020-2022 Saint Petersburg State University
//* All Rights Reserved
//* See file LICENSE for details.
//***************************************************************************

#ifdef SPADES_CUDA_SUPPORT

#include "gpu_kmer_kernels.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>

namespace gpu {
namespace kernels {

// Constants for DNA encoding
__constant__ uint8_t d_nucleotide_map[256];

// Initialize nucleotide mapping on device
__global__ void init_nucleotide_map() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < 256) {
        uint8_t value = 0xFF; // Invalid nucleotide
        
        switch (tid) {
            case 'A': case 'a': value = 0; break;
            case 'C': case 'c': value = 1; break;
            case 'G': case 'g': value = 2; break;
            case 'T': case 't': value = 3; break;
            default: value = 0xFF; break;
        }
        
        d_nucleotide_map[tid] = value;
    }
}

// Extract k-mers from a sequence
__global__ void extract_kmers_kernel(
    const char* __restrict__ sequences,
    const uint32_t* __restrict__ seq_offsets,
    const uint32_t* __restrict__ seq_lengths,
    uint64_t* __restrict__ kmers,
    uint32_t* __restrict__ kmer_counts,
    uint32_t num_sequences,
    uint32_t k,
    uint32_t max_kmers_per_seq
) {
    uint32_t seq_idx = blockIdx.x;
    uint32_t thread_idx = threadIdx.x;
    
    if (seq_idx >= num_sequences) return;
    
    uint32_t seq_offset = seq_offsets[seq_idx];
    uint32_t seq_length = seq_lengths[seq_idx];
    
    if (seq_length < k) {
        if (thread_idx == 0) {
            kmer_counts[seq_idx] = 0;
        }
        return;
    }
    
    uint32_t num_kmers = seq_length - k + 1;
    uint32_t kmers_to_process = min(num_kmers, max_kmers_per_seq);
    
    // Shared memory for temporary kmer storage
    extern __shared__ uint64_t shared_kmers[];
    
    uint32_t valid_kmers = 0;
    
    // Process k-mers in parallel within block
    for (uint32_t kmer_idx = thread_idx; kmer_idx < kmers_to_process; kmer_idx += blockDim.x) {
        uint64_t kmer = 0;
        bool valid = true;
        
        // Extract k-mer starting at position kmer_idx
        for (uint32_t i = 0; i < k; ++i) {
            char nucleotide = sequences[seq_offset + kmer_idx + i];
            uint8_t encoded = d_nucleotide_map[static_cast<uint8_t>(nucleotide)];
            
            if (encoded == 0xFF) {
                valid = false;
                break;
            }
            
            kmer = (kmer << 2) | encoded;
        }
        
        if (valid) {
            // Compute canonical k-mer (lexicographically smaller of kmer and reverse complement)
            uint64_t rc_kmer = reverse_complement(kmer, k);
            uint64_t canonical_kmer = min(kmer, rc_kmer);
            
            shared_kmers[thread_idx] = canonical_kmer;
            valid_kmers++;
        }
    }
    
    __syncthreads();
    
    // Write valid k-mers to global memory
    if (thread_idx == 0) {
        uint32_t output_offset = seq_idx * max_kmers_per_seq;
        uint32_t write_count = 0;
        
        for (uint32_t i = 0; i < blockDim.x && write_count < valid_kmers; ++i) {
            if (i < kmers_to_process) {
                kmers[output_offset + write_count] = shared_kmers[i];
                write_count++;
            }
        }
        
        kmer_counts[seq_idx] = write_count;
    }
}

// Count k-mers using hash table
__global__ void count_kmers_kernel(
    const uint64_t* __restrict__ kmers,
    const uint32_t* __restrict__ kmer_counts,
    const uint32_t* __restrict__ kmer_offsets,
    uint64_t* __restrict__ hash_keys,
    uint32_t* __restrict__ hash_values,
    uint32_t* __restrict__ hash_status,
    uint32_t num_sequences,
    uint32_t hash_table_size
) {
    uint32_t seq_idx = blockIdx.x;
    uint32_t thread_idx = threadIdx.x;
    
    if (seq_idx >= num_sequences) return;
    
    uint32_t kmer_offset = kmer_offsets[seq_idx];
    uint32_t num_kmers = kmer_counts[seq_idx];
    
    // Process k-mers for this sequence
    for (uint32_t i = thread_idx; i < num_kmers; i += blockDim.x) {
        uint64_t kmer = kmers[kmer_offset + i];
        
        // Insert into hash table using linear probing
        uint32_t hash = hash_function(kmer) % hash_table_size;
        
        while (true) {
            uint32_t status = atomicCAS(&hash_status[hash], HASH_EMPTY, HASH_INSERTING);
            
            if (status == HASH_EMPTY) {
                // Successfully claimed empty slot
                hash_keys[hash] = kmer;
                hash_values[hash] = 1;
                __threadfence();
                hash_status[hash] = HASH_OCCUPIED;
                break;
            } else if (status == HASH_OCCUPIED && hash_keys[hash] == kmer) {
                // K-mer already exists, increment count
                atomicAdd(&hash_values[hash], 1);
                break;
            } else {
                // Collision, try next slot
                hash = (hash + 1) % hash_table_size;
            }
        }
    }
}

// Optimized k-mer extraction with memory coalescing
__global__ void extract_kmers_coalesced_kernel(
    const char* __restrict__ sequences,
    const uint32_t* __restrict__ seq_info, // packed: offset (24 bits) + length (8 bits)
    uint64_t* __restrict__ kmers,
    uint32_t* __restrict__ kmer_positions,
    uint32_t num_sequences,
    uint32_t k,
    uint32_t max_output_kmers
) {
    uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_threads = gridDim.x * blockDim.x;
    
    __shared__ uint64_t temp_kmers[256]; // Shared memory for temporary storage
    __shared__ uint32_t temp_positions[256];
    
    uint32_t local_kmer_count = 0;
    uint32_t write_offset = 0;
    
    // Process sequences in chunks
    for (uint32_t seq_chunk = 0; seq_chunk < num_sequences; seq_chunk += total_threads) {
        uint32_t seq_idx = seq_chunk + global_tid;
        
        if (seq_idx < num_sequences) {
            uint32_t packed_info = seq_info[seq_idx];
            uint32_t seq_offset = packed_info & 0xFFFFFF; // Lower 24 bits
            uint32_t seq_length = packed_info >> 24;      // Upper 8 bits
            
            if (seq_length >= k) {
                uint32_t num_kmers = seq_length - k + 1;
                
                // Extract k-mers from this sequence
                for (uint32_t pos = 0; pos < num_kmers; ++pos) {
                    uint64_t kmer = 0;
                    bool valid = true;
                    
                    // Build k-mer
                    for (uint32_t i = 0; i < k; ++i) {
                        char nucleotide = sequences[seq_offset + pos + i];
                        uint8_t encoded = d_nucleotide_map[static_cast<uint8_t>(nucleotide)];
                        
                        if (encoded == 0xFF) {
                            valid = false;
                            break;
                        }
                        
                        kmer = (kmer << 2) | encoded;
                    }
                    
                    if (valid && local_kmer_count < 256) {
                        // Store k-mer and position in shared memory
                        uint64_t rc_kmer = reverse_complement(kmer, k);
                        temp_kmers[local_kmer_count] = min(kmer, rc_kmer);
                        temp_positions[local_kmer_count] = (seq_idx << 16) | pos; // Pack seq_idx and position
                        local_kmer_count++;
                    }
                }
            }
        }
        
        __syncthreads();
        
        // Write to global memory in coalesced manner
        if (threadIdx.x == 0) {
            write_offset = atomicAdd(&kmer_positions[0], local_kmer_count); // Use position[0] as counter
        }
        __syncthreads();
        
        if (write_offset + local_kmer_count <= max_output_kmers) {
            for (uint32_t i = threadIdx.x; i < local_kmer_count; i += blockDim.x) {
                kmers[write_offset + i] = temp_kmers[i];
                kmer_positions[write_offset + i + 1] = temp_positions[i]; // +1 to skip counter at [0]
            }
        }
        
        local_kmer_count = 0;
        __syncthreads();
    }
}

// Hash function for k-mers
__device__ __forceinline__ uint32_t hash_function(uint64_t kmer) {
    // MurmurHash3-like function
    kmer ^= kmer >> 33;
    kmer *= 0xff51afd7ed558ccdULL;
    kmer ^= kmer >> 33;
    kmer *= 0xc4ceb9fe1a85ec53ULL;
    kmer ^= kmer >> 33;
    return static_cast<uint32_t>(kmer);
}

// Reverse complement function
__device__ __forceinline__ uint64_t reverse_complement(uint64_t kmer, uint32_t k) {
    uint64_t rc = 0;
    
    for (uint32_t i = 0; i < k; ++i) {
        uint64_t nucleotide = kmer & 3;
        uint64_t complement = 3 - nucleotide; // A<->T (0<->3), C<->G (1<->2)
        rc = (rc << 2) | complement;
        kmer >>= 2;
    }
    
    return rc;
}

// Sort k-mers using CUB radix sort
__global__ void sort_kmers_setup_kernel(
    const uint64_t* __restrict__ input_kmers,
    uint64_t* __restrict__ output_kmers,
    uint32_t* __restrict__ indices,
    uint32_t num_kmers
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_kmers) {
        output_kmers[tid] = input_kmers[tid];
        indices[tid] = tid;
    }
}

// Host interface functions
void initialize_gpu_nucleotide_map(cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(1);
    
    init_nucleotide_map<<<grid, block, 0, stream>>>();
}

void launch_extract_kmers_kernel(
    const char* sequences,
    const uint32_t* seq_offsets,
    const uint32_t* seq_lengths,
    uint64_t* kmers,
    uint32_t* kmer_counts,
    uint32_t num_sequences,
    uint32_t k,
    uint32_t max_kmers_per_seq,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(num_sequences);
    
    size_t shared_mem_size = block.x * sizeof(uint64_t);
    
    extract_kmers_kernel<<<grid, block, shared_mem_size, stream>>>(
        sequences, seq_offsets, seq_lengths, kmers, kmer_counts, 
        num_sequences, k, max_kmers_per_seq
    );
}

void launch_count_kmers_kernel(
    const uint64_t* kmers,
    const uint32_t* kmer_counts,
    const uint32_t* kmer_offsets,
    uint64_t* hash_keys,
    uint32_t* hash_values,
    uint32_t* hash_status,
    uint32_t num_sequences,
    uint32_t hash_table_size,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(num_sequences);
    
    count_kmers_kernel<<<grid, block, 0, stream>>>(
        kmers, kmer_counts, kmer_offsets, hash_keys, hash_values, hash_status,
        num_sequences, hash_table_size
    );
}

void launch_extract_kmers_coalesced_kernel(
    const char* sequences,
    const uint32_t* seq_info,
    uint64_t* kmers,
    uint32_t* kmer_positions,
    uint32_t num_sequences,
    uint32_t k,
    uint32_t max_output_kmers,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((num_sequences + block.x - 1) / block.x);
    
    extract_kmers_coalesced_kernel<<<grid, block, 0, stream>>>(
        sequences, seq_info, kmers, kmer_positions, 
        num_sequences, k, max_output_kmers
    );
}

} // namespace kernels
} // namespace gpu

#endif // SPADES_CUDA_SUPPORT 