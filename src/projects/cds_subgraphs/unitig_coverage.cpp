//***************************************************************************
//* Copyright (c) 2023-2024 SPAdes team
//* Copyright (c) 2018-2022 Saint Petersburg State University
//* All Rights Reserved
//* See file LICENSE for details.
//***************************************************************************

#include "profile_storage.hpp"

#include "alignment/kmer_mapper.hpp"
#include "alignment/sequence_mapper.hpp"
#include "configs/config_struct.hpp"
#include "io/dataset_support/dataset_readers.hpp"
#include "library/library_data.hpp"
#include "library/library.hpp"
#include "pipeline/graph_pack_helpers.h"
#include "pipeline/sequence_mapper_gp_api.hpp"
#include "projects/mts/contig_abundance.hpp"
#include "toolchain/edge_label_helper.hpp"
#include "toolchain/utils.hpp"
#include "utils/segfault_handler.hpp"

#include "version.hpp"

#include <clipp/clipp.h>
#include <unordered_map>
#include <numeric>
#include <sys/types.h>

using namespace debruijn_graph;

typedef io::DataSet<config::LibraryData> DataSet;
typedef io::SequencingLibrary<config::LibraryData> SequencingLib;

static io::ReadStreamList<io::SingleRead>
single_easy_readers_for_libs(DataSet& dataset_info,
                             const std::vector<size_t>& libs,
                             bool followed_by_rc = true,
                             bool including_paired_reads = true,
                             bool handle_Ns = true,
                             io::OffsetType offset_type = io::PhredOffset) {
    VERIFY(!libs.empty());
    io::ReadStreamList<io::SingleRead> streams;
    for (auto l_id : libs) {
        streams.push_back(io::single_easy_reader(dataset_info[l_id],
                                             followed_by_rc,
                                             including_paired_reads, handle_Ns, offset_type));
    }
    return streams;
}

static void Run(const std::filesystem::path &graph_path, const std::string &dataset_desc, size_t K,
         const std::filesystem::path &profiles_fn, size_t nthreads, const std::filesystem::path &tmpdir) {
    DataSet dataset;
    dataset.load(dataset_desc);

    graph_pack::GraphPack gp(K, tmpdir, dataset.lib_count());
    auto &graph = gp.get<Graph>();

    INFO("Loading de Bruijn graph from " << graph_path);
    omnigraph::GraphElementFinder<Graph> element_finder(graph);
    gp.get_mutable<KmerMapper<Graph>>().Attach();
    io::EdgeLabelHelper<Graph> label_helper(element_finder,
                                            toolchain::LoadGraphPack(gp,graph_path));

    config::init_libs(dataset, nthreads, tmpdir);

    EnsureBasicMapping(gp);

    std::vector<size_t> libs(dataset.lib_count());
    std::iota(libs.begin(), libs.end(), 0);

    auto single_readers = single_easy_readers_for_libs(dataset, libs,
            /*followed by rc*/true, /*including paired*/true);

    size_t sample_cnt = dataset.lib_count();
    debruijn_graph::coverage_profiles::EdgeProfileStorage profile_storage(graph, sample_cnt);

    profile_storage.Fill(single_readers, *MapperInstance(gp));

    std::ofstream os(profiles_fn);
    profile_storage.Save(os, label_helper.edge_naming_f());
}

struct gcfg {
    gcfg()
        : k(21),
          nthreads(omp_get_max_threads() / 2 + 1)
    {}

    unsigned k;
    std::filesystem::path file;
    std::string graph;
    std::filesystem::path tmpdir;
    std::filesystem::path outfile;
    unsigned nthreads;
};

static void process_cmdline(int argc, char **argv, gcfg &cfg) {
  using namespace clipp;

  std::string file;
  std::string tmpdir;
  std::string outfile;

  auto cli = (
      file << value("dataset description (in YAML)"),
      cfg.graph << value("graph (in GFA)"),
      outfile << value("output filename"),
      (option("-k") & integer("value", cfg.k)) % "k-mer length to use",
      (option("-t", "--threads") & integer("value", cfg.nthreads)) % "# of threads to use",
      (option("--tmpdir") & value("dir", tmpdir)) % "scratch directory to use"
  );

  auto result = parse(argc, argv, cli);
  if (!result) {
      std::cout << make_man_page(cli, argv[0]);
      exit(1);
  }

  cfg.file = file;
  cfg.tmpdir = tmpdir.empty() ? "tmp" : tmpdir;
  cfg.outfile = outfile.empty() ? "-" : outfile;
}

int main(int argc, char** argv) {
    utils::segfault_handler sh;
    gcfg cfg;

    process_cmdline(argc, argv, cfg);

    toolchain::create_console_logger();
    START_BANNER("Computing unitig coverage profiles across a list of samples");

    try {
        unsigned nthreads = cfg.nthreads;
        unsigned k = cfg.k;
        std::filesystem::path tmpdir = cfg.tmpdir;

        create_directory(tmpdir);

        INFO("K-mer length set to " << k);

        nthreads = std::min(nthreads, (unsigned) omp_get_max_threads());
        // Inform OpenMP runtime about this :)
        omp_set_num_threads((int) nthreads);
        INFO("# of threads to use: " << nthreads);

        Run(cfg.graph, cfg.file, k, cfg.outfile, nthreads, tmpdir);
    } catch (const std::string &s) {
        std::cerr << s << std::endl;
        return EINTR;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return EINTR;
    }
}
