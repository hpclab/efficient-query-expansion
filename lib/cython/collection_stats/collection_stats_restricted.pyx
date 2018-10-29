# distutils: language = c++

include "_collection_stats.pyx"

cdef class PyCollectionStatsRestricted(_PyCollectionStats):
    def update(self, PyCollectionStatsRestricted other):
        self.c_collection_stats.update(dereference(other.c_collection_stats))

    @staticmethod
    def load(str filename):
        return PyCollectionStatsRestricted(filename=filename)

    @staticmethod
    def loads(str dump_str):
        return PyCollectionStatsRestricted(dump_str=dump_str)

cdef class PyCollectionStatsRestrictedFiller(_PyCollectionStatsFiller):
    def add_restriction(self, size_t first, second=None, third=None):
        assert first is not None
        assert third is None or second is not None

        cdef size_t _second
        cdef size_t _third

        if second is None:
            self.c_collection_stats_filler.add_restriction(first)
        elif third is None:
            _second = second
            self.c_collection_stats_filler.add_restriction(first, _second)
        else:
            _second = second
            _third = third
            self.c_collection_stats_filler.add_restriction(first, _second, _third)
