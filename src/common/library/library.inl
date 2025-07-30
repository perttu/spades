template<class Data>
using Library = io::SequencingLibrary<Data>;

namespace llvm { namespace yaml {
template <class Data>
struct SequenceTraits<std::vector<Library<Data> >>  {
    static size_t size(IO &, std::vector<Library<Data> > &seq) {
        return seq.size();
    }
    static Library<Data>&
    element(IO &, std::vector<Library<Data>> &seq, size_t index) {
        if (index >= seq.size())
            seq.resize(index+1);
        return seq[index];
    }
};

template <class Data>
struct SequenceTraits<io::DataSet<Data>>  {
    static size_t size(IO &, io::DataSet<Data> &seq) {
        return seq.lib_count();
    }
    static Library<Data>&
    element(IO &, io::DataSet<Data> &seq, size_t index) {
        if (index >= seq.lib_count())
            seq.push_back(Library<Data>());
        return seq[index];
    }
};


template<class Data>
void MappingTraits<Library<Data>>::mapping(yaml::IO &io, Library<Data> &lib) {
    lib.yamlize(io);
}

template<class Data>
StringRef MappingTraits<Library<Data>>::validate(yaml::IO &io, Library<Data> &lib) {
    // We use such crazy API for validate() since we don't want to pull
    // llvm::StringRef into library.hpp.
    llvm::StringRef res;
    lib.validate(io, res);

    return res;
}

}}

template<class Data>
void io::SequencingLibrary<Data>::yamlize(llvm::yaml::IO &io) {
    // First, load the "common stuff"
    SequencingLibraryBase::yamlize(io);
    io.mapOptional("data", data_);
}

template<class Data>
void io::SequencingLibrary<Data>::validate(llvm::yaml::IO &io, llvm::StringRef &res) {
    // Simply ask base class to validate for us
    SequencingLibraryBase::validate(io, res);
}

template<class Data>
void io::DataSet<Data>::save(const std::filesystem::path &filename) {
    std::error_code EC;
    llvm::raw_fd_ostream ofs(filename.c_str(), EC, llvm::sys::fs::OpenFlags::F_Text);
    llvm::yaml::Output yout(ofs);
    yout << libraries_;
}

template<class Data>
void io::DataSet<Data>::load(const std::filesystem::path &filename) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> Buf = MemoryBuffer::getFile(filename.c_str());
    if (!Buf) {
        std::cerr << "Failed to load file " << filename;
        throw;
    }

    yaml::Input yin(*Buf.get());
    yin >> libraries_;

    if (yin.error()) {
        std::cerr << "Failed to load file " << filename;
        throw;
    }
    
    std::filesystem::path input_dir = filename.parent_path();

    for (unsigned i = 0; i < libraries_.size(); ++i) {
        auto &lib = libraries_[i];
        if (lib.number() == -1u) {
            lib.set_number(i);
        }

        lib.update_relative_reads_filenames(input_dir);
    }
}
