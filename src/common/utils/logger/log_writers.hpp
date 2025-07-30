//***************************************************************************
//* Copyright (c) 2023-2024 SPAdes team
//* Copyright (c) 2015-2022 Saint Petersburg State University
//* Copyright (c) 2011-2014 Saint Petersburg Academic University
//* All Rights Reserved
//* See file LICENSE for details.
//***************************************************************************

#pragma once

#include "logger.hpp"
#include <mutex>

namespace logging {

struct console_writer : public writer {
    void write_msg(double time, size_t cmem, size_t max_rss, level l, const std::filesystem::path& file, size_t line_num,
                   const char *source, const char *msg);
};

class file_writer : public writer {
public:
    file_writer(const std::string &filename) : fout(filename) {}

    void write_msg(double time, size_t cmem, size_t max_rss, level l, const std::filesystem::path& file, size_t line_num,
                   const char *source, const char *msg);

private:
    std::ofstream fout;
};

class mutex_writer : public writer {
    std::mutex writer_mutex_;
    std::shared_ptr<writer> writer_;
public:
    mutex_writer(std::shared_ptr<writer> writer) : writer_(writer) {}

    void write_msg(double time, size_t cmem, size_t max_rss, level l, const std::filesystem::path& file, size_t line_num,
                   const char *source, const char *msg);
};

} // logging
