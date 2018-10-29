# distutils: language=c++

import collections
cimport cython
from libcpp cimport bool
from libcpp.vector cimport vector
from cython.operator cimport dereference

from pattern_matching.pattern_matcher cimport PatternMatcher, PyPatternMatcher
from pattern_matching.pattern_matcher cimport PatternMatches, PyPatternMatches

cdef extern from "<iostream>" namespace "std":
    cdef cppclass istream:
        istream()
    cdef cppclass ostream:
        ostream()

cdef extern from "<sstream>" namespace "std":
    cdef cppclass ostringstream(ostream):
        ostringstream()
        string str()

    cdef cppclass istringstream(istream):
        istringstream()
        istringstream(string)


StatsTerm = collections.namedtuple('StatsTerm', ["df", "tf", "tf_square"])
StatsTermPair = collections.namedtuple('StatsTermPair', ["df", "window_df", "window_tf", "window_tf_square", "window_min_dist"])
StatsTermTriple = collections.namedtuple('StatsTermTriple', ["df", "window_df", "window_tf", "window_tf_square", "window_min_dist"])


cdef class _PyCollectionStats:
    def __cinit__(self, distance_t window_size_co_occ2=12, distance_t window_size_co_occ3=15, str filename=None, str dump_str=None):
        if filename and dump_str:
            raise ValueError("filename and dump cannot be set at the same time")
        cdef string _filename
        cdef istringstream ss

        if filename or dump_str:
            if filename:
                _filename = filename
                with nogil:
                    self.c_collection_stats = CollectionStats[uint32_t, CSF_DISABLE_UNWINDOWED_TYPE, CS_RESTRICTED_TYPE].load(_filename)
            else:
                ss = istringstream(dump_str)
                with nogil:
                    self.c_collection_stats = CollectionStats[uint32_t, CSF_DISABLE_UNWINDOWED_TYPE, CS_RESTRICTED_TYPE].loads(&ss)
        else:
            self.c_collection_stats = new CollectionStats[uint32_t, CSF_DISABLE_UNWINDOWED_TYPE, CS_RESTRICTED_TYPE](window_size_co_occ2, window_size_co_occ3)

    def __dealloc__(self):
        del self.c_collection_stats

    def clear(self):
        self.c_collection_stats.clear()

    def get_stats_term(self, uint32_t pattern_id):
        cdef StatsKey stats = self.c_collection_stats.get_stats_key(pattern_id)
        return StatsTerm(stats.document_frequency, stats.frequency, stats.frequency_square)

    def get_stats_term_pair(self, uint32_t pattern_id1, uint32_t pattern_id2):
        cdef StatsKeyPair stats = self.c_collection_stats.get_stats_key_pair(pattern_id1, pattern_id2)
        return StatsTermPair(stats.document_frequency, stats.window_document_frequency, stats.window_frequency, stats.window_frequency_square, stats.window_min_dist)

    def get_stats_term_triple(self, uint32_t pattern_id1, uint32_t pattern_id2, uint32_t pattern_id3):
        cdef StatsKeyTriple stats = self.c_collection_stats.get_stats_key_triple(pattern_id1, pattern_id2, pattern_id3)
        return StatsTermTriple(stats.document_frequency, stats.window_document_frequency, stats.window_frequency, stats.window_frequency_square, stats.window_min_dist)

    def get_num_docs(self):
        return self.c_collection_stats.get_num_docs()

    def get_term_frequency_sum(self):
        return self.c_collection_stats.get_key_frequency_sum()

    def get_term_pair_window_co_occ_sum(self):
        return self.c_collection_stats.get_key_pair_window_co_occ_sum()

    def get_term_triple_window_co_occ_sum(self):
        return self.c_collection_stats.get_key_triple_window_co_occ_sum()

    def get_num_terms(self):
        return self.c_collection_stats.get_num_keys()

    def get_num_term_pairs(self):
        return self.c_collection_stats.get_num_key_pairs()

    def get_num_term_triples(self):
        return self.c_collection_stats.get_num_key_triples()

    def update(self, _PyCollectionStats other):
        self.c_collection_stats.update(dereference(other.c_collection_stats))

    def dump(self, str filename):
        self.c_collection_stats.dump(filename)

    def dumps(self):
        cdef ostringstream ss
        self.c_collection_stats.dumps(&ss)
        return ss.str()

#    @staticmethod
#    def load(str filename):
#        return _PyCollectionStats(filename=filename)
#
#    @staticmethod
#    def loads(str dump_str):
#        return _PyCollectionStats(dump_str=dump_str)


cdef class _PyCollectionStatsFiller:
    def __cinit__(
            self,
            _PyCollectionStats collection_stats,
            PyPatternMatcher pattern_matcher,
            size_t buffer_size_in_bytes,
            uint32_t num_threads,
            uint32_t queue_max_size,
    ):
        self.c_collection_stats_filler = new CollectionStatsFiller[uint32_t, CSF_DISABLE_UNWINDOWED_TYPE, CS_RESTRICTED_TYPE, CSF_BUFFERED_WORKER_TYPE, CSF_BUFFERED_COLLECTOR_TYPE](
            collection_stats.c_collection_stats,
            pattern_matcher.c_matcher,
            buffer_size_in_bytes,
            num_threads,
            queue_max_size
        )

    def __dealloc__(self):
        del self.c_collection_stats_filler

    @cython.boundscheck(False)
    def update(
            self,
            list doc_fields
    ):
        if len(doc_fields) == 0:
            return
        cdef vector[string] c_doc_fields

        # fill the vector
        for i in range(len(doc_fields)):
            c_doc_fields.push_back(doc_fields[i])

        # update call
        self.c_collection_stats_filler.update(
            c_doc_fields
        )

    def flush(self):
        self.c_collection_stats_filler.flush()
