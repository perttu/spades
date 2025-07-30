//***************************************************************************
//* Copyright (c) 2023-2024 SPAdes team
//* Copyright (c) 2018-2022 Saint Petersburg State University
//* All Rights Reserved
//* See file LICENSE for details.
//***************************************************************************

#pragma once

#include "assembly_graph/core/graph.hpp"
#include "io/utils/edge_namer.hpp"

#include <memory>
#include <string>
#include <ostream>

namespace io {

class FastgWriter {
  protected:
    typedef debruijn_graph::DeBruijnGraph Graph;

public:
    FastgWriter(const Graph &graph,
                const std::filesystem::path &fn,
                io::EdgeNamingF<Graph> edge_naming_f = io::BasicNamingF<Graph>())
            : graph_(graph), fn_(fn),
              short_namer_(graph_),
              extended_namer_(graph_, edge_naming_f, "", "'") {
    }

    void WriteSegmentsAndLinks();

  protected:
    const Graph &graph_;
    const std::filesystem::path &fn_;
    io::CanonicalEdgeHelper<Graph> short_namer_;
    io::CanonicalEdgeHelper<Graph> extended_namer_;
};

}

