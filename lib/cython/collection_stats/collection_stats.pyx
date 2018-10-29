# distutils: language = c++

include "_collection_stats.pyx"

cdef class PyCollectionStats(_PyCollectionStats):
    def update(self, PyCollectionStats other):
        self.c_collection_stats.update(dereference(other.c_collection_stats))

    @staticmethod
    def load(str filename):
        return PyCollectionStats(filename=filename)

    @staticmethod
    def loads(str dump_str):
        return PyCollectionStats(dump_str=dump_str)

cdef class PyCollectionStatsFiller(_PyCollectionStatsFiller):
    pass
