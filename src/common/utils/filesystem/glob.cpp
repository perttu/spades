//***************************************************************************
//* Copyright (c) 2023-2024 SPAdes team
//* Copyright (c) 2018-2022 Saint Petersburg State University
//* All Rights Reserved
//* See file LICENSE for details.
//***************************************************************************

#include "glob.hpp"

#include <glob.h>

#include <cstring>
#include <stdexcept>

namespace fs {

// https://stackoverflow.com/questions/8401777/simple-glob-in-c-on-unix-system
// TODO add only_dirs/only_files option?
std::vector<std::filesystem::path> glob(const std::string &pattern) {
    glob_t glob_result;
    ::memset(&glob_result, 0, sizeof(glob_result));

    int return_value = ::glob(pattern.c_str(), GLOB_NOESCAPE | GLOB_NOSORT | GLOB_ERR | GLOB_MARK,
                              NULL, &glob_result);
    if (return_value == GLOB_NOMATCH) {
        ::globfree(&glob_result);
        return {};
    }

    if (return_value == GLOB_NOSPACE || return_value == GLOB_ABORTED) {
        ::globfree(&glob_result);
        const char *status = return_value == GLOB_NOSPACE ? "GLOB_NOSPACE" : "GLOB_ABORTED";
        throw std::runtime_error(std::string("::glob() failed with return_value ") + status);
    }

    std::vector<std::filesystem::path> filenames;
    for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
        filenames.emplace_back(glob_result.gl_pathv[i]);
    }

    ::globfree(&glob_result);
    return filenames;
}

}  // namespace fs
