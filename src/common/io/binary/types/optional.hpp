//***************************************************************************
//* Copyright (c) 2023-2024 SPAdes team
//* Copyright (c) 2019-2022 Saint Petersburg State University
//* All Rights Reserved
//* See file LICENSE for details.
//***************************************************************************

#pragma once

#include "common/io/binary/binary.hpp"
#include <optional>

namespace io {
namespace binary {
namespace impl {

template <typename T>
class Serializer<std::optional<T>, std::enable_if_t<io::binary::is_serializable<T>>> {
public:
    static void Write(std::ostream &os, const std::optional<T> &v) {
        if (v) {
            io::binary::BinWrite(os, true, *v);
        } else {
            io::binary::BinWrite(os, false);
        }
    }

    static void Read(std::istream &is, std::optional<T> &v) {
        auto present = io::binary::BinRead<bool>(is);
        if (present) {
            auto val = io::binary::BinRead<T>(is);
            v = std::make_optional(val);
        } else {
            v = std::nullopt;
        }
    }
};
}  // namespace impl
}  // namespace binary
}  // namespace io
