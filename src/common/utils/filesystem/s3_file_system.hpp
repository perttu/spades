//***************************************************************************
//* Copyright (c) 2023-2024 SPAdes team
//* Copyright (c) 2020-2022 Saint Petersburg State University
//* All Rights Reserved
//* See file LICENSE for details.
//***************************************************************************

#pragma once

#include <fstream>
#include <string>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <filesystem>
#include <memory>
#include <thread>
#include <atomic>

namespace fs {
namespace s3 {

struct S3Config {
    std::string bucket;
    std::string region = "us-east-1";
    std::string access_key;
    std::string secret_key;
    std::string endpoint_url; // for S3-compatible services
    std::filesystem::path local_cache_dir = "/tmp/spades_s3_cache";
    size_t cache_size_limit = 4ULL * 1024 * 1024 * 1024; // 4TB
    size_t chunk_size = 64 * 1024 * 1024; // 64MB chunks
    int max_parallel_downloads = 10;
    bool enable_prefetching = true;
};

struct CacheEntry {
    std::filesystem::path local_path;
    std::string s3_key;
    size_t size = 0;
    std::chrono::system_clock::time_point last_access;
    bool dirty = false;
    bool downloading = false;
    std::atomic<bool> ready{false};
};

class S3FileSystem {
private:
    S3Config config_;
    std::unordered_map<std::string, CacheEntry> cache_;
    std::mutex cache_mutex_;
    std::atomic<size_t> current_cache_size_{0};
    std::thread cache_manager_thread_;
    std::atomic<bool> shutdown_{false};

public:
    explicit S3FileSystem(const S3Config& config);
    ~S3FileSystem();

    // Main interface - extends existing fs::open_file
    std::ifstream open_file(const std::string& uri, 
                           std::ios_base::openmode mode = std::ios_base::in,
                           std::ios_base::iostate exception_bits = std::ios_base::failbit | std::ios_base::badbit);

    // Check if URI is S3
    static bool is_s3_uri(const std::string& uri);
    
    // Parse S3 URI into bucket and key
    static std::pair<std::string, std::string> parse_s3_uri(const std::string& uri);

    // Preload datasets for pipeline stages
    void preload_dataset(const std::vector<std::string>& s3_keys);
    
    // Background cache management
    void manage_cache();
    
    // Upload dirty files back to S3
    void sync_to_s3();
    
    // Get cache statistics
    struct CacheStats {
        size_t total_entries;
        size_t total_size;
        size_t hits;
        size_t misses;
    };
    CacheStats get_cache_stats() const;

private:
    // Internal methods
    std::filesystem::path ensure_cached(const std::string& s3_key);
    void download_file(const std::string& s3_key, const std::filesystem::path& local_path);
    void evict_oldest_entries(size_t needed_space);
    void start_cache_manager();
    void stop_cache_manager();
    
    // Background prefetching
    void prefetch_next_chunk(const std::string& s3_key, size_t offset);
    
    // S3 client operations (to be implemented with AWS SDK)
    bool s3_download(const std::string& key, const std::filesystem::path& local_path);
    bool s3_upload(const std::filesystem::path& local_path, const std::string& key);
    size_t s3_get_file_size(const std::string& key);
};

// Global S3 filesystem instance
extern std::unique_ptr<S3FileSystem> g_s3fs;

// Initialize S3 filesystem
void init_s3_filesystem(const S3Config& config);

// Cleanup S3 filesystem
void cleanup_s3_filesystem();

} // namespace s3
} // namespace fs 