from featurizer import Featurizer
import numpy as np
from numpy cimport ndarray
cimport numpy as np
cimport cython


class FeaturizerTags(Featurizer):
    def __init__(self, tags, *args, **kwargs):
        assert hasattr(tags, "__iter__") and all(isinstance(tag, str) for tag in tags)
        tags = sorted(set(tags), key=lambda tag: tag.lower())
        feature_names = tags + [
            "{}_{}".format(pref, tag)
            for pref in ["num_base_syn", "num_syn"]
            for tag in tags
        ]
        assert len(feature_names) == len(tags) * 3

        super(FeaturizerTags, self).__init__(feature_names, *args, **kwargs)
        self._tag_to_pos = dict((tag, i) for i, tag in enumerate(tags))

    def _transform_impl(
            self,
            base_repr, exp_repr, num_exp_terms,
            global_features, from_row, from_column
    ):
        _c_get_features(
            base_repr, exp_repr, num_exp_terms,
            global_features, from_row, from_column,
            self._tag_to_pos
        )


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _c_get_features(
        list base_repr,
        list exp_repr,
        int num_exp_terms,
        ndarray[np.float32_t, ndim=2] global_features,
        unsigned int from_row,
        unsigned int from_column,
        dict tag_to_pos
):
    cdef np.float32_t* it_features

    cdef int num_tags = len(tag_to_pos)
    cdef ndarray[np.uint32_t, ndim=1] synset_num_syn = np.empty(num_tags, dtype=np.uint32)
    cdef ndarray[np.uint32_t, ndim=1] synset_num_base_syn = np.empty(num_tags, dtype=np.uint32)
    cdef int c1 = from_column + num_tags
    cdef int c2 = c1 + num_tags
    cdef int c3 = c2 + num_tags
    cdef int pos = 0
    cdef int and_pos = 0, syn_pos = 0

    # reset the portion to avoid untouched portions
    global_features[from_row:from_row+num_exp_terms, from_column:from_column+num_tags] = 0.0

    # loop over the expansions
    cdef int it = from_row
    cdef int it_row
    for and_pos, and_query in enumerate(exp_repr):  # the query is the OR composition of different AND_QUERIES
        for syn_pos, synset in enumerate(and_query):  # the AND_QUERY is the AND composition of different SYNSET (CNF)
            # fill synset_num_base_syn
            synset_num_base_syn[:] = 0
            for term_tags in base_repr[and_pos][syn_pos]:
                if len(term_tags) > 1:
                    for tag in term_tags[1]:
                        synset_num_base_syn[tag_to_pos[tag]] += 1

            synset_num_syn[:] = 0
            for term_tags in synset:  # the SYNSET is the OR composition of different synonyms
                it_features = &global_features[it,from_column]
                it += 1

                if len(term_tags) > 1:
                    for tag in term_tags[1]:
                        pos = tag_to_pos[tag]
                        it_features[pos] = 1
                        synset_num_syn[pos] += 1

            for it_row in range(it-len(synset), it):
                global_features[it_row, c1:c2] = synset_num_base_syn
                global_features[it_row, c2:c3] = synset_num_syn
    # TODO: TO CHECK
