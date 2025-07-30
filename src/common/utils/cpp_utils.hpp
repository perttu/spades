//***************************************************************************
//* Copyright (c) 2023-2024 SPAdes team
//* Copyright (c) 2015-2022 Saint Petersburg State University
//* Copyright (c) 2011-2014 Saint Petersburg Academic University
//* All Rights Reserved
//* See file LICENSE for details.
//***************************************************************************

#pragma once

#include <cstddef>

namespace utils {

// arrays
template<class T, size_t N>
size_t array_size(T (&/*arr*/)[N]) {
    return N;
}

template<class T, size_t N>
T *array_end(T (&arr)[N]) {
    return &arr[N];
}

template<size_t EXPECTED_SIZE, class T, size_t N>
void check_array_size(T (&/*arr*/)[N]) {
    static_assert(EXPECTED_SIZE == N, "Unexpected array size");
}

template<class T>
T identity_function(const T &t) {
    return t;
}

template<typename Base, typename T>
inline bool instanceof(const T *ptr) {
    return dynamic_cast<const Base *>(ptr) != nullptr;
}

} // namespace utils
