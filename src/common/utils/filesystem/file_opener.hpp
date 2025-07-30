//***************************************************************************
//* Copyright (c) 2023-2024 SPAdes team
//* Copyright (c) 2020-2022 Saint Petersburg State University
//* All Rights Reserved
//* See file LICENSE for details.
//***************************************************************************

#pragma once 

#include <fstream>
#include <string>

namespace fs {

/// @returns an opened file stream
/// @throw std::ios_base::failure if the file does not exists
/// @note Be careful with std::ios_base::failbit and reading a file until the eof
template <class FileName>
std::ifstream open_file(FileName && file_name,
                       std::ios_base::openmode mode = std::ios_base::in,
                       std::ios_base::iostate exception_bits = std::ios_base::failbit | std::ios_base::badbit)
{
    std::string filename_str = std::string(file_name);
    
    // Check if this is an S3 URI
    if (filename_str.substr(0, 5) == "s3://") {
        // Use S3 filesystem if available
        #ifdef SPADES_S3_SUPPORT
        if (s3::g_s3fs) {
            return s3::g_s3fs->open_file(filename_str, mode, exception_bits);
        }
        #endif
        
        // Fall back to regular filesystem (will fail, but provides clear error)
        throw std::ios_base::failure("S3 support not enabled. Cannot open S3 file: " + filename_str);
    }
    
    // Regular filesystem
    std::ifstream file(std::forward<FileName>(file_name), mode);
    if (!file.is_open())
        throw std::ios_base::failure("Cannot open file '" + std::string(file_name) + '\'');
    file.exceptions(exception_bits);
    return file;
}

} // namespace fs
