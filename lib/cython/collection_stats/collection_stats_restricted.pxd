# distutils: language = c++

cdef extern from "CollectionStats.hpp":
    # fake import to fix the template to true
    cdef cppclass CSF_DISABLE_UNWINDOWED_TYPE "true":
        pass
    cdef cppclass CS_RESTRICTED_TYPE "true":
        pass
    cdef cppclass CSF_BUFFERED_WORKER_TYPE "false":
        pass
    cdef cppclass CSF_BUFFERED_COLLECTOR_TYPE "false":
        pass

include "_collection_stats.pxd"
