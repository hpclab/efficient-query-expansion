from featurizer import Featurizer

import numpy as np
from numpy cimport ndarray
cimport numpy as np
cimport cython

class FeaturizerTextual(Featurizer):
    def __init__(self, *args, **kwargs):
        super(FeaturizerTextual, self).__init__(feature_names=_feature_names, *args, **kwargs)

    def _transform_impl(
            self,
            base_repr, exp_repr, num_exp_terms,
            global_features, from_row, from_column
    ):
        # convert the two representations using the
        base_repr = [
            [[syn_tags[0].replace(" ", "") for syn_tags in synset]
             for synset in and_query] for and_query in base_repr
        ]
        exp_repr = [
            [[syn_tags[0].replace(" ", "") for syn_tags in synset]
             for synset in and_query] for and_query in exp_repr
        ]

        _c_get_features(
            base_repr, exp_repr, num_exp_terms,
            global_features, from_row, from_column
        )


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef size_t levenshtein(
    str string_1,
    str string_2,
    size_t len_1,
    size_t len_2
):
    """copyied from https://github.com/toastdriven/pylev/blob/master/pylev.py"""
    len_1 += 1
    len_2 += 1
    cdef ndarray[np.uint64_t, ndim=1] d = np.empty(len_1* len_2, dtype=np.uint64)
    cdef size_t i, j

    for i in range(len_1):
        d[i] = i
    for j in range(len_2):
        d[j * len_1] = j

    for j in range(1, len_2):
        for i in range(1, len_1):
            if string_1[i - 1] == string_2[j - 1]:
                d[i + j * len_1] = d[i - 1 + (j - 1) * len_1]
            else:
                d[i + j * len_1] = min(
                   d[i - 1 + j * len_1] + 1,        # deletion
                   d[i + (j - 1) * len_1] + 1,      # insertion
                   d[i - 1 + (j - 1) * len_1] + 1,  # substitution
                )

    return d[len_1 * len_2 - 1]


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
        unsigned int from_column
):
    cdef np.float32_t* it_features

    # temporary variables
    cdef size_t i_and, and_query_pos
    cdef size_t len_term, len_base, tmp_base_len, first_base_len, min_len, i
    cdef size_t edit_min_len, pref_min_len, suff_min_len
    cdef size_t edit, pref, suff
    cdef double pref_max_ratio, suff_max_ratio
    cdef size_t tmp_edit, tmp_pref, tmp_suff
    cdef double tmp_ratio_pref, tmp_ratio_suff
    cdef double tmp_len_ratio, edit_len_ratio, pref_len_ratio, suff_len_ratio
    cdef str base_term, term

    # loop over the expansions
    cdef int it = from_row
    for i_and, and_query in enumerate(exp_repr):  # the query is the OR composition of different AND_QUERIES
        for and_query_pos, synset in enumerate(and_query):  # the AND_QUERY is the AND composition of different SYNSET (CNF)
            base_synset = base_repr[i_and][and_query_pos]
            first_base_len = len(base_synset[0]) if len(base_synset) else 0

            for term in synset:  # the SYNSET is the OR composition of different synonyms
                it_features = &global_features[it,from_column]
                it += 1

                len_term = len(term)

                edit = len(term) + first_base_len  # all edit distances are lower than this one
                pref = suff = 0
                edit_min_len = pref_min_len = suff_min_len = 0
                pref_max_ratio = suff_max_ratio = 0
                edit_len_ratio = pref_len_ratio = suff_len_ratio = 0

                for base_term in base_synset:
                    tmp_base_len = len(base_term)
                    min_len = min(len_term, tmp_base_len)
                    tmp_len_ratio = 1.0 * len_term / tmp_base_len

                    # edit
                    tmp_edit = levenshtein(term, base_term, len_term, tmp_base_len)
                    if tmp_edit < edit:  # lower is better
                        edit = tmp_edit
                        edit_min_len = min_len
                        edit_len_ratio = tmp_len_ratio

                    tmp_pref = 0
                    for i in range(min_len):
                        if term[i] == base_term[i]:
                            tmp_pref += 1
                        else:
                            break
                    tmp_ratio_pref = 1.0 * tmp_pref / min_len
                    if tmp_ratio_pref > pref_max_ratio:  # higher is better
                        pref = tmp_pref
                        pref_min_len = min_len
                        pref_max_ratio = tmp_ratio_pref
                        pref_len_ratio = tmp_len_ratio

                    tmp_suff = 0
                    for i in range(1, min_len+1):
                        if term[len_term - i] == base_term[tmp_base_len - i]:
                            tmp_suff += 1
                        else:
                            break
                    tmp_ratio_suff = 1.0 * tmp_suff / min_len
                    if tmp_ratio_suff > suff_max_ratio:  # higher is better
                        suff = tmp_suff
                        suff_min_len = min_len
                        suff_max_ratio = tmp_ratio_suff
                        suff_len_ratio = tmp_len_ratio

                it_features[ 0] = len_term

                it_features[ 1] = edit
                it_features[ 2] = edit_min_len
                it_features[ 3] = edit_len_ratio

                it_features[ 4] = pref
                it_features[ 5] = pref_min_len
                it_features[ 6] = pref_max_ratio
                it_features[ 7] = pref_len_ratio

                it_features[ 8] = suff
                it_features[ 9] = suff_min_len
                it_features[10] = suff_max_ratio
                it_features[11] = suff_len_ratio

# this double check avoid mistakes
cdef int _num_features = 12
cdef tuple _feature_names = (
    "exp_length",

    "edit_dist",
    "edit_min_common_len",
    "edit_len_ratio_dist",

    "prefix_dist",
    "pref_min_common_len",
    "pref_max_ratio_dist",
    "pref_len_ratio_dist",

    "suffix_dist",
    "suff_min_common_len",
    "suff_max_ratio_dist",
    "suff_len_ratio_dist",
)

assert len(_feature_names) == _num_features
