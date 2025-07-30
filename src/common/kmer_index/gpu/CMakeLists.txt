# GPU k-mer index components
# This CMakeLists.txt is included only when SPADES_GPU_SUPPORT is enabled

if(SPADES_GPU_SUPPORT AND SPADES_CUDA_SUPPORT)
    project(gpu_kmer_index CXX CUDA)
    
    # Collect GPU sources
    set(GPU_KMER_SOURCES
        gpu_kmer_kernels.cu
        gpu_hash_table.cu
        gpu_kmer_index.cpp
    )
    
    set(GPU_KMER_HEADERS
        gpu_kmer_kernels.cuh
        gpu_hash_table.hpp
        gpu_kmer_index.hpp
    )
    
    # Create the GPU k-mer index library
    add_library(gpu_kmer_index STATIC ${GPU_KMER_SOURCES} ${GPU_KMER_HEADERS})
    
    # Set CUDA properties
    set_target_properties(gpu_kmer_index PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CXX_STANDARD 17
        CUDA_STANDARD 17
    )
    
    # Link CUDA libraries
    target_link_libraries(gpu_kmer_index 
        ${GPU_LIBRARIES}
        utils
    )
    
    # Include directories
    target_include_directories(gpu_kmer_index PRIVATE
        ${CMAKE_SOURCE_DIR}/src/common
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    
    # Export for other modules to use
    set(GPU_KMER_INDEX_LIBRARY gpu_kmer_index PARENT_SCOPE)
    
else()
    # Create empty library when GPU support is disabled
    add_library(gpu_kmer_index INTERFACE)
    set(GPU_KMER_INDEX_LIBRARY gpu_kmer_index PARENT_SCOPE)
endif() 