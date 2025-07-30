//***************************************************************************
//* Copyright (c) 2023-2024 SPAdes team
//* Copyright (c) 2021-2022 Saint Petersburg State University
//* All Rights Reserved
//* See file LICENSE for details.
//***************************************************************************

#pragma once

#include "binning.hpp"
#include "id_map.hpp"

#include "assembly_graph/core/graph.hpp"

#include <blaze/Forward.h>
#include <unordered_map>

namespace bin_stats {

class Binning;
struct EdgeLabels;

using SoftBinsAssignment = adt::id_map<EdgeLabels, debruijn_graph::EdgeId>;

class BinningAssignmentStrategy {
public:
    BinningAssignmentStrategy(bool allow_multiple = false)
            : allow_multiple_(allow_multiple) {}
    
    virtual void AssignEdgeBins(const SoftBinsAssignment& soft_bins_assignment,
                                Binning& bin_stats) const = 0;
    virtual blaze::CompressedVector<double> AssignScaffoldBins(const std::vector<debruijn_graph::EdgeId>& path,
                                                               const SoftBinsAssignment& soft_bins_assignment,
                                                               const Binning& bin_stats) const = 0;
    // FIXME: temporary return uint64_t, not BinId, until we refine cyclic deps
    virtual std::vector<uint64_t> ChooseMajorBins(const blaze::CompressedVector<double>& bins_weights,
                                                  const SoftBinsAssignment& soft_bins_assignment,
                                                  const Binning& bin_stats) const;
    virtual std::vector<uint64_t> ChooseMajorBins(const std::vector<debruijn_graph::EdgeId>& path,
                                                  const SoftBinsAssignment& soft_bins_assignment,
                                                  const Binning& bin_stats) const;

    virtual ~BinningAssignmentStrategy() = default;

  protected:
    bool allow_multiple_;
};
}
