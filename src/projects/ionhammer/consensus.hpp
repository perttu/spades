//***************************************************************************
//* Copyright (c) 2023-2024 SPAdes team
//* Copyright (c) 2015-2022 Saint Petersburg State University
//* Copyright (c) 2011-2014 Saint Petersburg Academic University
//* All Rights Reserved
//* See file LICENSE for details.
//***************************************************************************

#ifndef __HAMMER_IT_CONSENSUS_HPP__
#define __HAMMER_IT_CONSENSUS_HPP__

#include "HSeq.hpp"

#include <blaze/math/StaticMatrix.h>
#include <limits>

namespace hammer {
namespace iontorrent {

using ScoreMatrix = blaze::StaticMatrix<double, 4, 64>;

inline std::pair<hammer::HomopolymerRun, double> consensus(
    const ScoreMatrix &scores) {
  double inf = -std::numeric_limits<double>::infinity();

  double max = inf;
  uint8_t nucl = 0;
  uint8_t len = 1;
  for (uint8_t j = 0; j < 4; ++j)
    for (uint8_t k = 1; k < 64; ++k)
      if (scores(j, k) > max) {
        nucl = j;
        len = k;
        max = scores(j, k);
      }

  return std::make_pair(hammer::HomopolymerRun(nucl, len), max);
}

};  // namespace iontorrent
};  // namespace hammer

#endif  // __HAMMER_IT_CONSENSUS_HPP__
