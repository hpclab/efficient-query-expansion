from featurizer_collection_stats import FeaturizerCollectionStats
import numpy as np
from numpy cimport ndarray
cimport numpy as np
cimport cython

from libc.math cimport log2


class FeaturizerSigIR08(FeaturizerCollectionStats):
    def __init__(self, collection_stats, collection_stats_segment_to_segment_id, *args, **kwargs):
        super(FeaturizerSigIR08, self).__init__(
            feature_names=_feature_names,
            collection_stats_feature_fun=_c_get_features,
            collection_stats=collection_stats,
            collection_stats_segment_to_segment_id=collection_stats_segment_to_segment_id,
            *args, **kwargs
        )


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef float _c_fraction_log(size_t numerator, size_t denominator):
    assert 0 <= numerator <= denominator
    return log2(1.0 + 1.0 * numerator / (denominator if denominator > 0 else 1.0))


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
    object collection_stats
):
    cdef np.float32_t* it_features

    # temporary variables
    cdef list base_and_query
    cdef size_t i_and
    cdef size_t exp_id

    cdef size_t num_base_terms
    cdef size_t num_base_pairs
    cdef size_t exp_tf, exp_df
    cdef size_t sum_query_terms_tf
    cdef size_t exp_co_occ2
    cdef size_t exp_co_occ3
    cdef size_t exp_co_occ2_weighted

    cdef size_t collection_tf_sum = collection_stats.get_term_frequency_sum()

    # loop over the expansions
    cdef int it = from_row
    for i_and, and_query in enumerate(exp_repr):  # the query is the OR composition of different AND_QUERIES
        base_and_query = base_repr[i_and]

        # normalization factor
        sum_query_terms_tf = 0
        num_base_terms = 0
        for base_synset in base_and_query:
            num_base_terms += len(base_synset)
            for base_termid_term_tags in base_synset:
                sum_query_terms_tf += collection_stats.get_stats_term(base_termid_term_tags[0]).tf
        num_base_pairs = (num_base_terms) * (num_base_terms - 1) if num_base_terms >= 2 else 0

        for synset in and_query:  # the AND_QUERY is the AND composition of different SYNSET (CNF)
            for termid_term_tags in synset:  # the SYNSET is the OR composition of different synonyms
                it_features = &global_features[it,from_column]
                it += 1

                # exp term id and term frequency
                exp_id = termid_term_tags[0]
                stats_exp = collection_stats.get_stats_term(exp_id)
                exp_tf = stats_exp.tf
                exp_df = stats_exp.df

                # co_occ2, co_occ2 weighted
                exp_co_occ2 = 0
                exp_co_occ2_weighted = 0
                # compute the co_occ2 with all the query terms
                for base_synset in base_and_query:
                    for base_termid_term_tags in base_synset:
                        stats_term_pair = collection_stats.get_stats_term_pair(base_termid_term_tags[0], exp_id)
                        exp_co_occ2 += stats_term_pair.window_tf
                        exp_co_occ2_weighted += stats_term_pair.window_tf * stats_term_pair.window_min_dist

                # co_occ3
                exp_co_occ3 = 0
                # compute the co_occ2 with all the query pairs
                if exp_co_occ2 > 0 and num_base_pairs > 0:
                    for base_synset in base_and_query:
                        for base_termid_term_tags in base_synset:
                            for base_synset_2 in base_and_query:
                                for base_termid_term_tags_2 in base_synset_2:
                                    exp_co_occ3 += collection_stats.get_stats_term_triple(exp_id, base_termid_term_tags[0], base_termid_term_tags_2[0]).window_tf
                    exp_co_occ3 = exp_co_occ3 / 2

                # term distribution
                it_features[0] = _c_fraction_log(exp_tf, sum_query_terms_tf)
                # co_occ2
                it_features[1] = _c_fraction_log(exp_co_occ2, num_base_terms * sum_query_terms_tf)
                # co_occ3
                it_features[2] = _c_fraction_log(exp_co_occ3, num_base_pairs * sum_query_terms_tf)
                # co_occ2 weighted
                it_features[3] = _c_fraction_log(exp_co_occ2_weighted, exp_co_occ2)


# this double check avoid mistakes
cdef int _num_features = 4
cdef tuple _feature_names = (
    "term_distribution",
    "co_occ2",
    "co_occ3",
    "co_occ2_w"
)

assert len(_feature_names) == _num_features
