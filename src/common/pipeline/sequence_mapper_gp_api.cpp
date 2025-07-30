//***************************************************************************
//* Copyright (c) 2023-2024 SPAdes team
//* Copyright (c) 2020-2022 Saint Petersburg State University
//* All Rights Reserved
//* See file LICENSE for details.
//***************************************************************************

#include "sequence_mapper_gp_api.hpp"

#include "alignment/edge_index.hpp"

namespace debruijn_graph {
std::shared_ptr<BasicSequenceMapper<Graph, EdgeIndex<Graph>>> MapperInstance(const graph_pack::GraphPack &gp) {
    return std::make_shared<BasicSequenceMapper<Graph, EdgeIndex<Graph>>>(gp.get<Graph>(),
            gp.get<EdgeIndex<Graph>>(),
            gp.get<KmerMapper<Graph>>());
}

std::shared_ptr<BasicSequenceMapper<Graph, EdgeIndex<Graph>>> MapperInstance(const graph_pack::GraphPack &gp,
                                                                             const EdgeIndex<Graph> &index) {
    return std::make_shared<BasicSequenceMapper<Graph, EdgeIndex<Graph>>>(gp.get<Graph>(),
            index,
            gp.get<KmerMapper<Graph>>());
}
}

