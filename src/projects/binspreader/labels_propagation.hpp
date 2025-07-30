//***************************************************************************
//* Copyright (c) 2023-2024 SPAdes team
//* Copyright (c) 2021-2022 Saint Petersburg State University
//* All Rights Reserved
//* See file LICENSE for details.
//***************************************************************************

#pragma once

#include "binning.hpp"
#include "binning_refiner.hpp"

#include "id_map.hpp"

namespace bin_stats {

class LabelsPropagation : public BinningRefiner {
 public:
    using FinalIteration = bool;
    using AlphaAssignment = adt::id_map<double, debruijn_graph::EdgeId>;

    LabelsPropagation(const debruijn_graph::Graph& g,
                      const binning::LinkIndex &links,
                      const AlphaAssignment &labeled_alpha,
                      const std::unordered_set<debruijn_graph::EdgeId> &nonpropagating_edges,
                      double eps, unsigned niter);

    SoftBinsAssignment RefineBinning(const SoftBinsAssignment &origin_state) const override;

 private:
    void EqualizeConjugates(SoftBinsAssignment& state) const;

    FinalIteration PropagationIteration(SoftBinsAssignment& new_state,
                                        const SoftBinsAssignment& cur_state,
                                        const SoftBinsAssignment& origin_state,
                                        const AlphaAssignment &alpha,
                                        unsigned iteration_step) const;

//    FullAlphaAssignment InitAlpha(const SoftBinsAssignment &origin_state) const;
//    void AlphaPropagationIteration(adt::id_map<double, debruijn_graph::EdgeId> &new_ealpha,
//                                   const adt::id_map<double, debruijn_graph::EdgeId> &ealpha,
//                                   const SoftBinsAssignment& origin_state,
//                                   unsigned iteration_step) const;

    AlphaAssignment labeled_alpha_;
    std::unordered_set<debruijn_graph::EdgeId> nonpropagating_edges_;
    const double eps_;
    const unsigned niter_;

    adt::id_map<double, debruijn_graph::EdgeId> rdeg_;
    adt::id_map<double, debruijn_graph::EdgeId> rweight_;
};
}
