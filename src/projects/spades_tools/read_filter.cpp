//***************************************************************************
//* Copyright (c) 2023-2024 SPAdes team
//* Copyright (c) 2015-2022 Saint Petersburg State University
//* Copyright (c) 2011-2014 Saint Petersburg Academic University
//* All Rights Reserved
//* See file LICENSE for details.
//***************************************************************************

#include "io/dataset_support/read_converter.hpp"
#include "io/reads/file_read_flags.hpp"
#include "io/reads/osequencestream.hpp"
#include "io/reads/coverage_filtering_read_wrapper.hpp"
#include "io/reads/longest_valid_wrapper.hpp"

#include "utils/parallel/openmp_wrapper.h"
#include "utils/logger/log_writers.hpp"
#include "utils/segfault_handler.hpp"
#include "kmer_index/kmer_counting.hpp"
#include "utils/filesystem/temporary.hpp"

#include "adt/cyclichash.hpp"
#include "adt/cqf.hpp"

#include "threadpool/threadpool.hpp"

#include "version.hpp"

#include <clipp/clipp.h>
#include <sys/types.h>
#include <string>

using namespace std;
typedef rolling_hash::SymmetricCyclicHash<> SeqHasher;

void create_console_logger() {
    using namespace logging;

    logger *lg = create_logger("");
    lg->add_writer(std::make_shared<console_writer>());
    attach_logger(lg);
}

namespace read_filter {
struct Args {
    unsigned thr = 2, k = 21;
    std::filesystem::path dataset_desc, workdir;
    unsigned nthreads = (omp_get_max_threads() / 2);
    bool drop_names = false;
    bool drop_quality = false;
};
}

void process_cmdline(int argc, char **argv, read_filter::Args &args) {
    using namespace clipp;
    bool print_help = false;
    std::string dataset_desc, workdir = ".";

    auto cli = (
        (option("-k", "--kmer") & integer("value", args.k)) % "K-mer length",
        (option("-c", "--cov") & integer("value", args.thr)) % "Median kmer count threshold (read pairs, s.t. kmer count median for BOTH reads LESS OR EQUAL to this value will be ignored)",
        (required("-d", "--dataset") & value("yaml", dataset_desc)) % "Dataset description (in YAML)",
        (option("-t", "--threads") & integer("value", args.nthreads)) % "# of threads to use",
        (option("-o", "--outdir") & value("dir", workdir)) %  "Output directory to use",
        (option("--drop-names").set(args.drop_names)) % "Drop read names and quality (makes everything faster)",
        (option("--drop-quality").set(args.drop_quality)) % "Drop read quality (makes everything faster)",
        (option("-h", "--help").set(print_help)) % "Show help"
    );

    auto result = parse(argc, argv, cli);
    if (!result || print_help) {
        std::cout << make_man_page(cli, argv[0]).prepend_section("DESCRIPTION", " Kmer count read filter");
        if (print_help) {
            exit(0);
        } else {
            exit(1);
        }
    }
    args.dataset_desc = dataset_desc;
    args.workdir = workdir;
}

template<class IS, class OS, class Filter>
void filter_reads(IS &input, OS &output, const Filter& filter, unsigned buffer_size, unsigned nthreads) {
    std::vector<typename OS::ReadT> reads_buffer(buffer_size);
    std::vector<bool> need_to_out(buffer_size);
    std::vector<unsigned> chunk_start(nthreads), chunk_end(nthreads);

    size_t read_count = 0, retained = 0;
    while (!input.eof()) {
        unsigned reads_cnt = 0;
        while (!input.eof() && reads_cnt < reads_buffer.size()) {
            input >> reads_buffer[reads_cnt];
            ++reads_cnt; ++read_count;
        }

        unsigned reads_per_thread = reads_cnt/nthreads;
        chunk_start[0] = 0;
        chunk_end[0] = reads_per_thread;
        for (unsigned i = 1; i < nthreads; ++i) {
            chunk_start[i] = chunk_end[i - 1];
            chunk_end[i] = chunk_start[i] + reads_per_thread;
        }
        chunk_end[nthreads - 1] = reads_cnt;

#       pragma omp parallel for
        for (unsigned i = 0; i < nthreads; ++i) {
            for (unsigned j = chunk_start[i]; j < chunk_end[i]; ++j) {
                typename OS::ReadT longest_valid_read = reads_buffer[j];
                io::LongestValid(longest_valid_read);

                need_to_out[j] = filter(longest_valid_read);
            }
        }

        for (size_t i = 0; i < reads_cnt; ++i) {
            if (need_to_out[i]) {
                output << reads_buffer[i];
                retained += 1;
            }
            need_to_out[i] = false;
        }

        VERBOSE_POWER(read_count, " reads processed");
    }
    INFO("Total " << read_count << " reads processed, " << retained << " reads left after filtering");
}

int main(int argc, char* argv[]) {
    typedef qf::cqf CQFKmerFilter;
    //typedef CyclicHash<64, uint8_t, NDNASeqHash<uint8_t>> SeqHasher;
    utils::perf_counter pc;

    srand(42);
    srandom(42);
    try {
        std::vector<std::string> input;

        read_filter::Args args;
        process_cmdline(argc, argv, args);

        create_console_logger();

        args.nthreads = spades_set_omp_threads(args.nthreads);

        INFO("Starting kmer count based read filtering, built from " << version::refspec() << ", git revision " << version::gitrev());

        INFO("K-mer length set to " << args.k);
        INFO("# of threads to use: " << args.nthreads);
        INFO("Maximum # of threads to use (adjusted due to OMP capabilities): " << args.nthreads);

        io::DataSet<debruijn_graph::config::LibraryData> dataset;
        dataset.load(args.dataset_desc);
        create_directory(args.workdir);
        auto tmpdir = fs::tmp::make_temp_dir(args.workdir, "binreads");
        debruijn_graph::config::init_libs(dataset, args.nthreads, tmpdir->dir());

        std::unique_ptr<ThreadPool::ThreadPool> pool;

        if (args.nthreads > 1) {
            pool = std::make_unique<ThreadPool::ThreadPool>(args.nthreads);
        }

        for (size_t i = 0; i < dataset.lib_count(); ++i) {
            io::ReadConverter::ConvertToBinary(dataset[i], pool.get());
        }

        std::vector<size_t> libs(dataset.lib_count());
        std::iota(libs.begin(), libs.end(), 0);
        io::BinarySingleStreams single_readers = io::single_binary_readers_for_libs(dataset, libs,
                                                                                    /*followed by rc*/false, /*including paired*/true);
        INFO("Estimating kmer cardinality");
        SeqHasher hasher(args.k);

        size_t kmers_cnt_est = kmers::EstimateCardinalityUpperBound(args.k, single_readers, hasher);

        CQFKmerFilter cqf(kmers_cnt_est);
        INFO("Filling kmer coverage");
        kmers::FillCoverageHistogram(cqf, args.k, hasher, single_readers, args.thr + 1);
        INFO("Kmer coverage filled");

        const unsigned FILTER_READS_BUFF_SIZE = 1 << 20;

        io::DataSet<debruijn_graph::config::LibraryData> outdataset;
        for (size_t i = 0; i < dataset.lib_count(); ++i) {
            auto outlib = dataset[i];
            outlib.clear();
            outlib.data().clear();

            INFO("Filtering library " << i);
            dataset[i].set_orientation(io::LibraryOrientation::Undefined);
            io::FileReadFlags flags(io::PhredOffset, !args.drop_names, !args.drop_names, !args.drop_quality, false);
            if (dataset[i].has_paired()) {
                io::PairedStream paired_reads_stream =
                        io::paired_easy_reader(dataset[i], /*followed by rc*/false, /*insert size*/0, false /* use orientation */, false /* handle Ns */,
                                               flags);
                io::CoverageFilter<io::PairedRead, SeqHasher> filter(args.k, hasher, cqf, args.thr + 1);
                // FIXME: we cannot use unique_ptr here as OFasta / OFastq streams do not have common base class :(
                if (args.drop_names || args.drop_quality) {
                    std::filesystem::path left = to_string(i + 1) + ".1.fasta";
                    std::filesystem::path right = to_string(i + 1) + ".2.fasta";

                    io::OFastaPairedStream ostream(args.workdir / left, args.workdir / right);
                    filter_reads(paired_reads_stream, ostream, filter, FILTER_READS_BUFF_SIZE, args.nthreads);
                    outlib.push_back_paired(left, right);
                } else {
                    std::filesystem::path left = args.workdir / (to_string(i + 1) + ".1.fastq");
                    std::filesystem::path right = args.workdir / (to_string(i + 1) + ".2.fastq");

                    io::OFastqPairedStream ostream(left, right);
                    filter_reads(paired_reads_stream, ostream, filter, FILTER_READS_BUFF_SIZE, args.nthreads);
                    outlib.push_back_paired(left, right);
                }
            }

            if (dataset[i].has_single()) {
                io::SingleStream single_reads_stream =
                        io::single_easy_reader(dataset[i], /*followed_by_rc*/ false, /*including_paired_reads*/ false,  /* handle Ns */ false,
                                               flags);
                io::CoverageFilter<io::SingleRead, SeqHasher> filter(args.k, hasher, cqf, args.thr + 1);
                // FIXME: we cannot use unique_ptr here as OFasta / OFastq streams do not have common base class :(
                if (args.drop_names || args.drop_quality) {
                    std::filesystem::path single = to_string(i + 1) + ".s.fasta";
                    io::OFastqReadStream ostream(args.workdir / single);
                    filter_reads(single_reads_stream, ostream, filter, FILTER_READS_BUFF_SIZE, args.nthreads);
                    outlib.push_back_single(single);
                } else {
                    std::filesystem::path single = args.workdir / (to_string(i + 1) + ".s.fastq");
                    io::OFastqReadStream ostream(single);
                    filter_reads(single_reads_stream, ostream, filter, FILTER_READS_BUFF_SIZE, args.nthreads);
                    outlib.push_back_single(single);
                }
            }

            if (dataset[i].has_merged()) {
                io::SingleStream single_reads_stream =
                        io::merged_easy_reader(dataset[i], /*followed_by_rc*/ false, /*handle_Ns*/ false,
                                               flags);
                io::CoverageFilter<io::SingleRead, SeqHasher> filter(args.k, hasher, cqf, args.thr + 1);
                // FIXME: we cannot use unique_ptr here as OFasta / OFastq streams do not have common base class :(
                if (args.drop_names || args.drop_quality) {
                    std::filesystem::path merged = args.workdir / (to_string(i + 1) + ".m.fasta");
                    io::OFastaReadStream ostream(merged);
                    filter_reads(single_reads_stream, ostream, filter, FILTER_READS_BUFF_SIZE, args.nthreads);
                    outlib.push_back_merged(merged);
                } else {
                    std::filesystem::path merged = to_string(i + 1) + ".m.fastq";
                    io::OFastqReadStream ostream(args.workdir / merged);
                    filter_reads(single_reads_stream, ostream, filter, FILTER_READS_BUFF_SIZE, args.nthreads);
                    outlib.push_back_merged(merged);
                }
            }

            outdataset.push_back(outlib);
        }
        INFO("Filtering finished");
        std::filesystem::path fname = args.workdir / "dataset.yaml";
        INFO("Saving filtered dataset description to " << fname);
        outdataset.save(fname);
    } catch (std::string const &s) {
        std::cerr << s;
        return EINTR;
    }

    return 0;
}
