//***************************************************************************
//* Copyright (c) 2023-2024 SPAdes team
//* Copyright (c) 2015-2022 Saint Petersburg State University
//* Copyright (c) 2011-2014 Saint Petersburg Academic University
//* All Rights Reserved
//* See file LICENSE for details.
//***************************************************************************

#pragma once

#include "assembly_graph/core/graph.hpp"
#include "io/reads/osequencestream.hpp"

#include <string>

namespace debruijn_graph {

inline void OutputEdgeSequences(const Graph &g, const std::string &contigs_output) {
    std::filesystem::path contigs_output_filename = contigs_output + ".fasta";
    INFO("Outputting contigs to " << contigs_output_filename);
    io::osequencestream_cov oss(contigs_output_filename);

    for (EdgeId e: g.canonical_edges()) {
        oss << g.coverage(e);
        oss << g.EdgeNucls(e).str();
    }
}

inline void OutputEdgesByID(const Graph &g, const std::string &contigs_output) {
    std::filesystem::path contigs_output_filename = contigs_output + ".fasta";
    INFO("Outputting contigs to " << contigs_output_filename);
    io::OFastaReadStream oss(contigs_output_filename);
    for (EdgeId e: g.canonical_edges()) {
        std::string s = g.EdgeNucls(e).str();
        oss << io::SingleRead(io::MakeContigId(g.int_id(e), s.size(), g.coverage(e), "EDGE"), s);
    }
}
} // namespace debruijn_graph

