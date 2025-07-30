//***************************************************************************
//* Copyright (c) 2023-2024 SPAdes team
//* Copyright (c) 2020-2022 Saint Petersburg State University
//* All Rights Reserved
//* See file LICENSE for details.
//***************************************************************************

#pragma once

#include <string>
#include <filesystem>

namespace debruijn_graph {
namespace config {

struct s3_config {
    // S3 connection settings
    std::string bucket;
    std::string region = "us-east-1";
    std::string access_key;
    std::string secret_key;
    std::string endpoint_url; // for S3-compatible services like MinIO
    
    // Cache settings
    std::filesystem::path local_cache_dir = "/tmp/spades_s3_cache";
    size_t cache_size_limit = 4ULL * 1024 * 1024 * 1024; // 4TB default
    size_t chunk_size = 64 * 1024 * 1024; // 64MB chunks
    
    // Performance settings
    int max_parallel_downloads = 10;
    int max_parallel_uploads = 5;
    bool enable_prefetching = true;
    bool enable_compression = true;
    
    // Pipeline-specific settings
    bool preload_reads = true;
    bool preload_reference = true;
    bool cache_intermediate_results = false;
    
    // Timeout settings
    int connect_timeout_seconds = 60;
    int request_timeout_seconds = 300;
    int retry_attempts = 3;
    
    // Debug settings
    bool enable_s3_logging = false;
    std::string s3_log_level = "info";
    
    // Validation
    bool validate() const {
        if (bucket.empty()) {
            return false;
        }
        if (access_key.empty() || secret_key.empty()) {
            return false;
        }
        if (cache_size_limit == 0) {
            return false;
        }
        return true;
    }
    
    // Convert to fs::s3::S3Config
    fs::s3::S3Config to_s3_config() const {
        fs::s3::S3Config config;
        config.bucket = bucket;
        config.region = region;
        config.access_key = access_key;
        config.secret_key = secret_key;
        config.endpoint_url = endpoint_url;
        config.local_cache_dir = local_cache_dir;
        config.cache_size_limit = cache_size_limit;
        config.chunk_size = chunk_size;
        config.max_parallel_downloads = max_parallel_downloads;
        config.enable_prefetching = enable_prefetching;
        return config;
    }
};

} // namespace config
} // namespace debruijn_graph 