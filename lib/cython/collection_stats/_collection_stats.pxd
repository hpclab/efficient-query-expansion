from libc.stdint cimport uint16_t, uint32_t, uint64_t
from libcpp.utility cimport pair
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

cimport pattern_matching.pattern_matcher as pattern_matcher
from pattern_matching.pattern_matcher cimport PatternMatcher, PatternMatches

ctypedef uint64_t       key_frequency_t
ctypedef uint32_t       document_frequency_t
ctypedef uint16_t       distance_t

cdef extern from "<iostream>" namespace "std":
    cdef cppclass istream
    cdef cppclass ostream


cdef extern from "CollectionStats.hpp":
    cdef cppclass KeyPair[T]:
        KeyPair (const T&, const T&)
        KeyPair (const KeyPair[T]&)

        const T& first() const
        const T& second() const

    cdef cppclass KeyTriple[T]:
        KeyTriple (const T&, const T&, const T&)
        KeyTriple (const KeyTriple[T]&)

        const T& first() const
        const T& second() const
        const T& third() const

    cdef cppclass StatsKey:
        StatsKey()
        StatsKey(const StatsKey&)

        document_frequency_t document_frequency
        key_frequency_t frequency
        key_frequency_t frequency_square

    cdef cppclass StatsKeyPair:
        StatsKeyPair()
        StatsKeyPair(const StatsKeyPair&)

        document_frequency_t document_frequency;
        document_frequency_t window_document_frequency;
        key_frequency_t window_frequency;
        key_frequency_t window_frequency_square;
        distance_t window_min_dist;

    cdef cppclass StatsKeyTriple:
        StatsKeyTriple()
        StatsKeyTriple(const StatsKeyTriple&)

        document_frequency_t document_frequency;
        document_frequency_t window_document_frequency;
        key_frequency_t window_frequency;
        key_frequency_t window_frequency_square;
        distance_t window_min_dist;


    cdef cppclass CollectionStats[T, BU, BR]:
        const distance_t window_size_key_pairs_co_occ
        const distance_t window_size_key_triples_co_occ

        CollectionStats ()
        CollectionStats (distance_t, distance_t)

        void                                                        clear()

        const StatsKey                                              get_stats_key(const T &) except +
        const StatsKeyPair                                          get_stats_key_pair(const T &, const T &) except +
        const StatsKeyTriple                                        get_stats_key_triple(const T &, const T &, const T &) except +

        document_frequency_t                                        get_num_docs() const
        key_frequency_t                                             get_key_frequency_sum() const
        key_frequency_t                                             get_key_pair_window_co_occ_sum() const
        key_frequency_t                                             get_key_triple_window_co_occ_sum() const

        size_t                                                      get_num_keys() const
        size_t                                                      get_num_key_pairs() const
        size_t                                                      get_num_key_triples() const

        void                                                        update(const CollectionStats[T, BU, BR] &) except +

        void                                                        dump(const string &) except +
        void                                                        dumps(ostream *) except +

        @staticmethod
        CollectionStats[T, BU, BR] *                                load(const string &) nogil except +
        @staticmethod
        CollectionStats[T, BU, BR] *                                loads(istream *) nogil except +


    cdef cppclass CollectionStatsFiller[T, BU, BR, BW, BC]:

        CollectionStatsFiller (CollectionStats*, PatternMatcher*, size_t, uint32_t, uint32_t)

        void                                                        add_restriction(const T&)
        void                                                        add_restriction(const T&, const T&)
        void                                                        add_restriction(const T&, const T&, const T&)

        void                                                        update(vector[string])
        void                                                        flush()


cdef class _PyCollectionStats:
    cdef CollectionStats[uint32_t, CSF_DISABLE_UNWINDOWED_TYPE, CS_RESTRICTED_TYPE] * c_collection_stats

cdef class _PyCollectionStatsFiller:
    cdef CollectionStatsFiller[uint32_t, CSF_DISABLE_UNWINDOWED_TYPE, CS_RESTRICTED_TYPE, CSF_BUFFERED_WORKER_TYPE, CSF_BUFFERED_COLLECTOR_TYPE] * c_collection_stats_filler
