//***************************************************************************
//* Copyright (c) 2023-2024 SPAdes team
//* Copyright (c) 2021-2022 Saint Petersburg State University
//* All Rights Reserved
//* See file LICENSE for details.
//***************************************************************************

#pragma once

#include "binning.hpp"

#include "assembly_graph/core/graph.hpp"

#include "library/library.hpp"
#include "library/library_data.hpp"

typedef io::DataSet<debruijn_graph::config::LibraryData> DataSet;
typedef io::SequencingLibrary<debruijn_graph::config::LibraryData> SequencingLib;

namespace binning {

void SplitAndWriteReads(const debruijn_graph::Graph &graph,
                        SequencingLib &lib,
                        const bin_stats::Binning &binning,
                        const bin_stats::SoftBinsAssignment& edge_soft_labels,
                        const bin_stats::BinningAssignmentStrategy& assignment_strategy,
                        const std::filesystem::path &work_dir,
                        const std::filesystem::path &prefix,
                        unsigned nthreads,
                        const double bin_weight_threshold);

}
