############################################################################
# Copyright (c) 2023-2024 SPAdes team
# Copyright (c) 2018-2022 Saint Petersburg State University
# All Rights Reserved
# See file LICENSE for details.
############################################################################

project(mph_index CXX)

add_library(mph_index STATIC kmer_mph/kmer_buckets.cpp)

# Add GPU k-mer index support if enabled
if(SPADES_GPU_SUPPORT)
    add_subdirectory(gpu)
    target_link_libraries(mph_index ${GPU_KMER_INDEX_LIBRARY})
endif()
