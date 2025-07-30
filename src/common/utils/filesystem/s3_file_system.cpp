//***************************************************************************
//* Copyright (c) 2023-2024 SPAdes team
//* Copyright (c) 2020-2022 Saint Petersburg State University
//* All Rights Reserved
//* See file LICENSE for details.
//***************************************************************************

#include "s3_file_system.hpp"
#include "utils/logger/logger.hpp"
#include "utils/filesystem/file_opener.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <regex>
#include <queue>

namespace fs {
namespace s3 {

// Global S3 filesystem instance
std::unique_ptr<S3FileSystem> g_s3fs;

S3FileSystem::S3FileSystem(const S3Config& config) 
    : config_(config) {
    
    // Create cache directory
    std::filesystem::create_directories(config_.local_cache_dir);
    
    // Start cache manager thread
    start_cache_manager();
    
    INFO("S3 FileSystem initialized with cache dir: " << config_.local_cache_dir);
}

S3FileSystem::~S3FileSystem() {
    stop_cache_manager();
    sync_to_s3();
}

bool S3FileSystem::is_s3_uri(const std::string& uri) {
    return uri.substr(0, 5) == "s3://";
}

std::pair<std::string, std::string> S3FileSystem::parse_s3_uri(const std::string& uri) {
    if (!is_s3_uri(uri)) {
        throw std::invalid_argument("Not an S3 URI: " + uri);
    }
    
    // Remove s3:// prefix
    std::string path = uri.substr(5);
    
    // Find first slash
    size_t slash_pos = path.find('/');
    if (slash_pos == std::string::npos) {
        throw std::invalid_argument("Invalid S3 URI format: " + uri);
    }
    
    std::string bucket = path.substr(0, slash_pos);
    std::string key = path.substr(slash_pos + 1);
    
    return {bucket, key};
}

std::ifstream S3FileSystem::open_file(const std::string& uri, 
                                     std::ios_base::openmode mode,
                                     std::ios_base::iostate exception_bits) {
    
    if (!is_s3_uri(uri)) {
        // Fall back to regular file system
        return fs::open_file(uri, mode, exception_bits);
    }
    
    auto [bucket, key] = parse_s3_uri(uri);
    
    // Ensure file is cached locally
    std::filesystem::path local_path = ensure_cached(key);
    
    // Update access time
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (cache_.find(key) != cache_.end()) {
            cache_[key].last_access = std::chrono::system_clock::now();
        }
    }
    
    // Open local cached file
    std::ifstream file(local_path, mode);
    if (!file.is_open()) {
        throw std::ios_base::failure("Cannot open S3 file '" + uri + "' (cached as '" + local_path.string() + "')");
    }
    file.exceptions(exception_bits);
    
    return file;
}

std::filesystem::path S3FileSystem::ensure_cached(const std::string& s3_key) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    auto it = cache_.find(s3_key);
    if (it != cache_.end() && it->second.ready) {
        return it->second.local_path;
    }
    
    // File not in cache, need to download
    std::filesystem::path local_path = config_.local_cache_dir / 
        std::filesystem::path(s3_key).filename();
    
    // Ensure unique filename
    int counter = 1;
    std::filesystem::path original_path = local_path;
    while (std::filesystem::exists(local_path)) {
        std::string stem = original_path.stem().string();
        std::string ext = original_path.extension().string();
        local_path = original_path.parent_path() / (stem + "_" + std::to_string(counter) + ext);
        counter++;
    }
    
    // Create cache entry
    CacheEntry entry;
    entry.local_path = local_path;
    entry.s3_key = s3_key;
    entry.last_access = std::chrono::system_clock::now();
    entry.downloading = true;
    
    cache_[s3_key] = entry;
    
    // Download in background
    std::thread download_thread([this, s3_key, local_path]() {
        try {
            download_file(s3_key, local_path);
            
            std::lock_guard<std::mutex> lock(cache_mutex_);
            auto it = cache_.find(s3_key);
            if (it != cache_.end()) {
                it->second.downloading = false;
                it->second.ready = true;
                it->second.size = std::filesystem::file_size(local_path);
                current_cache_size_ += it->second.size;
            }
        } catch (const std::exception& e) {
            ERROR("Failed to download S3 file " << s3_key << ": " << e.what());
            
            std::lock_guard<std::mutex> lock(cache_mutex_);
            cache_.erase(s3_key);
        }
    });
    download_thread.detach();
    
    return local_path;
}

void S3FileSystem::download_file(const std::string& s3_key, const std::filesystem::path& local_path) {
    INFO("Downloading S3 file: " << s3_key << " -> " << local_path);
    
    // TODO: Implement actual S3 download using AWS SDK
    // For now, create a placeholder file
    std::ofstream file(local_path);
    file << "# Placeholder for S3 file: " << s3_key << std::endl;
    file.close();
    
    // Simulate download time
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void S3FileSystem::preload_dataset(const std::vector<std::string>& s3_keys) {
    INFO("Preloading " << s3_keys.size() << " S3 files");
    
    for (const auto& key : s3_keys) {
        ensure_cached(key);
    }
}

void S3FileSystem::manage_cache() {
    while (!shutdown_) {
        std::this_thread::sleep_for(std::chrono::seconds(30));
        
        std::lock_guard<std::mutex> lock(cache_mutex_);
        
        // Check if we need to evict entries
        if (current_cache_size_ > config_.cache_size_limit) {
            size_t needed_space = current_cache_size_ - config_.cache_size_limit;
            evict_oldest_entries(needed_space);
        }
        
        // Clean up completed downloads
        for (auto it = cache_.begin(); it != cache_.end();) {
            if (!it->second.downloading && !it->second.ready) {
                it = cache_.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void S3FileSystem::evict_oldest_entries(size_t needed_space) {
    // Create vector of entries sorted by last access time
    std::vector<std::pair<std::string, std::chrono::system_clock::time_point>> entries;
    for (const auto& [key, entry] : cache_) {
        if (entry.ready && !entry.dirty) {
            entries.emplace_back(key, entry.last_access);
        }
    }
    
    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) {
                  return a.second < b.second;
              });
    
    // Evict oldest entries until we have enough space
    size_t freed_space = 0;
    for (const auto& [key, _] : entries) {
        if (freed_space >= needed_space) break;
        
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            freed_space += it->second.size;
            std::filesystem::remove(it->second.local_path);
            cache_.erase(it);
        }
    }
    
    current_cache_size_ -= freed_space;
    INFO("Evicted " << freed_space << " bytes from S3 cache");
}

void S3FileSystem::sync_to_s3() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    for (auto& [key, entry] : cache_) {
        if (entry.dirty && entry.ready) {
            // TODO: Implement S3 upload
            INFO("Syncing dirty file to S3: " << key);
            entry.dirty = false;
        }
    }
}

S3FileSystem::CacheStats S3FileSystem::get_cache_stats() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    CacheStats stats{};
    stats.total_entries = cache_.size();
    stats.total_size = current_cache_size_;
    
    // TODO: Track hits/misses
    stats.hits = 0;
    stats.misses = 0;
    
    return stats;
}

void S3FileSystem::start_cache_manager() {
    cache_manager_thread_ = std::thread(&S3FileSystem::manage_cache, this);
}

void S3FileSystem::stop_cache_manager() {
    shutdown_ = true;
    if (cache_manager_thread_.joinable()) {
        cache_manager_thread_.join();
    }
}

// Global functions
void init_s3_filesystem(const S3Config& config) {
    g_s3fs = std::make_unique<S3FileSystem>(config);
}

void cleanup_s3_filesystem() {
    if (g_s3fs) {
        g_s3fs->sync_to_s3();
        g_s3fs.reset();
    }
}

} // namespace s3
} // namespace fs 