from featurizer_collection_stats import FeaturizerCollectionStats
import numpy as np
from numpy cimport ndarray
cimport numpy as np
cimport cython

from libc.math cimport log2


class FeaturizerSigIR08extended(FeaturizerCollectionStats):
    def __init__(self, collection_stats, collection_stats_segment_to_segment_id, *args, **kwargs):
        super(FeaturizerSigIR08extended, self).__init__(
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
cdef void _c_get_cooccurrences(
    size_t ref_andpos,
    int ref_termid,
    list and_repr,
    collection_stats,
    ndarray[np.uint64_t, ndim=1] max_co_occ2_vec,
    ndarray[np.uint64_t, ndim=1] max_co_occ2_weighted_vec,
    ndarray[np.uint64_t, ndim=1] max_co_occ3_vec,
    ndarray[np.uint64_t, ndim=1] max_co_occ3_weighted_vec,
    char reset=True
):
    # temporary variables
    cdef size_t i
    cdef size_t andpos1, andpos2
    cdef size_t best_occ2_freq, best_occ2_gap, best_occ3_freq, best_occ3_gap
    cdef size_t occ_2_freq, occ_2_min_gap, occ_3_freq, occ_3_min_gap

    cdef char disable_co_occ3 = True

    # co-occ2
    i = 0
    for andpos1, synset1 in enumerate(and_repr):
        if andpos1 == ref_andpos:
            continue
        best_occ2_freq = best_occ2_gap = 0
        for termid_term_tags in synset1:
            stats_term_pair = collection_stats.get_stats_term_pair(termid_term_tags[0], ref_termid)
            occ_2_freq = stats_term_pair.window_tf
            occ_2_min_gap = stats_term_pair.window_min_dist
            if occ_2_freq > best_occ2_freq or (occ_2_freq == best_occ2_freq and occ_2_min_gap < best_occ2_gap):
                best_occ2_freq = occ_2_freq
                best_occ2_gap = occ_2_min_gap

        # update the the coocc2 vectors
        if reset or (best_occ2_freq > max_co_occ2_vec[i]) or (best_occ2_freq == max_co_occ2_vec[i] and best_occ2_freq * best_occ2_gap < max_co_occ2_weighted_vec[i]):
            max_co_occ2_vec[i] = best_occ2_freq
            max_co_occ2_weighted_vec[i] = best_occ2_freq * best_occ2_gap
        # enable the extraction of coocc3
        if best_occ2_freq > 0:
            disable_co_occ3 = False
        i += 1

    if disable_co_occ3:
        return

    i = 0
    for andpos1, synset1 in enumerate(and_repr):
        if andpos1 == ref_andpos:
            continue
        for andpos2, synset2 in enumerate(and_repr):
            if andpos2 <= andpos1 or andpos2 == ref_andpos:
                continue
            best_occ3_freq = best_occ3_gap = 0
            for termid_term_tags1 in synset1:
                for termid_term_tags2 in synset2:
                    stats_term_triple = collection_stats.get_stats_term_triple(ref_termid, termid_term_tags1[0], termid_term_tags2[0])
                    occ_3_freq = stats_term_triple.window_tf
                    occ_3_min_gap = stats_term_triple.window_min_dist
                    if occ_3_freq > best_occ3_freq or (occ_3_freq == best_occ3_freq and occ_3_min_gap < best_occ3_gap):
                        best_occ3_freq = occ_3_freq
                        best_occ3_gap = occ_3_min_gap

            # update the the coocc3 vector
            if reset or (best_occ3_freq > max_co_occ3_vec[i]) or (best_occ3_freq == max_co_occ3_vec[i] and best_occ3_freq * best_occ3_gap < max_co_occ3_weighted_vec[i]):
                max_co_occ3_vec[i] = best_occ3_freq
                max_co_occ3_weighted_vec[i] = best_occ3_freq * best_occ3_gap
            i += 1
    return
    # END


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _c_set_avg_min_max_features(
    np.float32_t * avg_exp, np.float32_t * avg_base, np.float32_t * avg_ratio,
    np.float32_t * min_exp, np.float32_t * min_base, np.float32_t * min_ratio,
    np.float32_t * max_exp, np.float32_t * max_base, np.float32_t * max_ratio,
    ndarray[np.uint64_t, ndim=1] exp_vec, ndarray[np.uint64_t, ndim=1] base_vec, size_t vec_size
):
    avg_exp[0] = avg_base[0] = avg_ratio[0] = 0
    min_exp[0] = min_base[0] = min_ratio[0] = 0
    max_exp[0] = max_base[0] = max_ratio[0] = 0

    if vec_size <= 0:
        return

    # set all the pointed values according to the first element
    avg_exp[0] = max_exp[0] = min_exp[0] = exp_vec[0]
    avg_base[0] = max_base[0] = min_base[0] = base_vec[0]
    avg_ratio[0] = min_ratio[0] = max_ratio[0] = 1.0 * exp_vec[0] / (base_vec[0] or 1)

    # update the values according to the remaining elements
    cdef size_t i
    cdef np.float32_t ratio
    for i in range(1, vec_size):
        avg_exp[0] += max_exp[0]
        if exp_vec[i] > max_exp[0]:
            max_exp[0] = exp_vec[i]
        elif exp_vec[i] < min_exp[0]:
            min_exp[0] = exp_vec[i]

        avg_base[0] += max_base[0]
        if base_vec[i] > max_base[0]:
            max_base[0] = base_vec[i]
        elif base_vec[i] < min_base[0]:
            min_base[0] = base_vec[i]

        ratio = 1.0 * exp_vec[i] / (base_vec[i] or 1)
        avg_ratio[0] += ratio
        if ratio > max_ratio[0]:
            max_ratio[0] = ratio
        elif ratio < min_ratio[0]:
            min_ratio[0] = ratio

    # here vec_size is greater than 0
    avg_exp[0] /= vec_size
    avg_base[0] /= vec_size
    avg_ratio[0] /= vec_size
    return


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

    max_len = max(len(and_query) for and_query in exp_repr)
    max_num_pairs = (max_len - 1) * (max_len - 2) / 2 if max_len >= 2 else 0

    # temporary variables
    cdef list base_and_query
    cdef size_t i_and, and_query_pos
    cdef size_t and_query_size
    cdef size_t num_co_occ2, num_co_occ3
    cdef size_t base_id
    cdef size_t exp_id

    cdef size_t base_tf = 0, base_df = 0
    cdef size_t exp_tf = 0, exp_df = 0
    cdef size_t tmp_tf = 0, tmp_df = 0

    cdef ndarray[np.uint64_t, ndim=1] base_co_occ2_vec = np.empty(max_len, dtype=np.uint64)
    cdef ndarray[np.uint64_t, ndim=1] exp_co_occ2_vec = np.empty(max_len, dtype=np.uint64)

    cdef ndarray[np.uint64_t, ndim=1] base_co_occ2_weighted_vec = np.empty(max_len, dtype=np.uint64)
    cdef ndarray[np.uint64_t, ndim=1] exp_co_occ2_weighted_vec = np.empty(max_len, dtype=np.uint64)

    cdef ndarray[np.uint64_t, ndim=1] base_co_occ3_vec = np.empty(max_num_pairs, dtype=np.uint64)
    cdef ndarray[np.uint64_t, ndim=1] exp_co_occ3_vec = np.empty(max_num_pairs, dtype=np.uint64)

    cdef ndarray[np.uint64_t, ndim=1] base_co_occ3_weighted_vec = np.empty(max_num_pairs, dtype=np.uint64)
    cdef ndarray[np.uint64_t, ndim=1] exp_co_occ3_weighted_vec = np.empty(max_num_pairs, dtype=np.uint64)

    # loop over the expansions
    cdef int it = from_row
    for i_and, and_query in enumerate(exp_repr):  # the query is the OR composition of different AND_QUERIES
        base_and_query = base_repr[i_and]
        and_query_size = len(base_and_query)
        num_co_occ2 = (and_query_size - 1)
        num_co_occ3 = (and_query_size - 1) * (and_query_size - 2) / 2 if and_query_size >= 2 else 0  # it is automatically 0 if and_query_size < 2

        for and_query_pos, synset in enumerate(and_query):  # the AND_QUERY is the AND composition of different SYNSET (CNF)

            base_tf = 0
            if and_query_size == 0:
                base_tf = 0
            else:
                termid_term_tags = base_and_query[and_query_pos][0]
                base_id = termid_term_tags[0]
                stats_term = collection_stats.get_stats_term(base_id)
                base_tf = stats_term.tf
                base_df = stats_term.df
                _c_get_cooccurrences(
                    and_query_pos, base_id, base_and_query, collection_stats,
                    base_co_occ2_vec, base_co_occ2_weighted_vec, base_co_occ3_vec, base_co_occ3_weighted_vec,
                    reset=True
                )

                for termid_term_tags in base_and_query[and_query_pos][1:]:
                    base_id = termid_term_tags[0]
                    stats_term = collection_stats.get_stats_term(base_id)
                    tmp_tf = stats_term.tf
                    tmp_df = stats_term.df
                    if tmp_tf > base_tf:
                        base_tf = tmp_tf
                    if tmp_df > base_df:
                        base_df = tmp_df
                    _c_get_cooccurrences(
                        and_query_pos, base_id, base_and_query, collection_stats,
                        base_co_occ2_vec, base_co_occ2_weighted_vec, base_co_occ3_vec, base_co_occ3_weighted_vec,
                        reset=False
                    )

            for termid_term_tags in synset:  # the SYNSET is the OR composition of different synonyms
                it_features = &global_features[it,from_column]
                it += 1

                # exp term id and term frequency
                exp_id = termid_term_tags[0]
                stats_term = collection_stats.get_stats_term(exp_id)
                exp_tf = stats_term.tf
                exp_df = stats_term.df

                # co_occ2, co_occ2 weighted
                _c_get_cooccurrences(
                    and_query_pos, exp_id, base_and_query, collection_stats,
                    exp_co_occ2_vec, exp_co_occ2_weighted_vec, exp_co_occ3_vec, exp_co_occ3_weighted_vec,
                    reset=True
                )

                # df
                it_features[ 0] = exp_df
                it_features[ 1] = base_df
                it_features[ 2] = 1.0 * exp_df / (base_df or 1)

                # tf
                it_features[ 3] = exp_tf
                it_features[ 4] = base_tf
                it_features[ 5] = 1.0 * exp_tf / (base_tf or 1)

                # co_occ2
                _c_set_avg_min_max_features(
                    avg_exp=  &it_features[ 6],
                    avg_base= &it_features[ 7],
                    avg_ratio=&it_features[ 8],
                    min_exp=  &it_features[ 9],
                    min_base= &it_features[10],
                    min_ratio=&it_features[11],
                    max_exp=  &it_features[12],
                    max_base= &it_features[13],
                    max_ratio=&it_features[14],
                    exp_vec=exp_co_occ2_vec,
                    base_vec=base_co_occ2_vec,
                    vec_size=num_co_occ2
                )

                # co_occ2 weighted
                _c_set_avg_min_max_features(
                    avg_exp=  &it_features[15],
                    avg_base= &it_features[16],
                    avg_ratio=&it_features[17],
                    min_exp=  &it_features[18],
                    min_base= &it_features[19],
                    min_ratio=&it_features[20],
                    max_exp=  &it_features[21],
                    max_base= &it_features[22],
                    max_ratio=&it_features[23],
                    exp_vec=exp_co_occ2_weighted_vec,
                    base_vec=base_co_occ2_weighted_vec,
                    vec_size=num_co_occ2
                )

                # co_occ3
                _c_set_avg_min_max_features(
                    avg_exp=  &it_features[24],
                    avg_base= &it_features[25],
                    avg_ratio=&it_features[26],
                    min_exp=  &it_features[27],
                    min_base= &it_features[28],
                    min_ratio=&it_features[29],
                    max_exp=  &it_features[30],
                    max_base= &it_features[31],
                    max_ratio=&it_features[32],
                    exp_vec=exp_co_occ3_vec,
                    base_vec=base_co_occ3_vec,
                    vec_size=num_co_occ3
                )

                # co_occ3 weighted
                _c_set_avg_min_max_features(
                    avg_exp=  &it_features[33],
                    avg_base= &it_features[34],
                    avg_ratio=&it_features[35],
                    min_exp=  &it_features[36],
                    min_base= &it_features[37],
                    min_ratio=&it_features[38],
                    max_exp=  &it_features[39],
                    max_base= &it_features[40],
                    max_ratio=&it_features[41],
                    exp_vec=exp_co_occ3_weighted_vec,
                    base_vec=base_co_occ3_weighted_vec,
                    vec_size=num_co_occ3
                )


# this double check avoid mistakes
cdef int _num_features = 42
cdef tuple _feature_names = (
    # df
    "exp_df",
    "base_df",
    "exp_df/base_df",
    # tf
    "exp_tf",
    "base_tf",
    "exp_tf/base_tf",
    # co_occ2
    "avg(exp_co_occ2)",
    "avg(base_co_occ2)",
    "avg(exp_co_occ2/base_co_occ2)",
    "min(exp_co_occ2)",
    "min(base_co_occ2)",
    "min(exp_co_occ2/base_co_occ2)",
    "max(exp_co_occ2)",
    "max(base_co_occ2)",
    "max(exp_co_occ2/base_co_occ2)",
    # co_occ2 weighted
    "avg(exp_co_occ2_w)",
    "avg(base_co_occ2_w)",
    "avg(exp_co_occ2_w/base_co_occ2_w))",
    "min(exp_co_occ2_w)",
    "min(base_co_occ2_w)",
    "min(exp_co_occ2_w/base_co_occ2_w)",
    "max(exp_co_occ2_w)",
    "max(base_co_occ2_w)",
    "max(exp_co_occ2_w/base_co_occ2_w)",
    # co_occ3
    "avg(exp_co_occ3)",
    "avg(base_co_occ3)",
    "avg(exp_co_occ3/base_co_occ3)",
    "min(exp_co_occ3)",
    "min(base_co_occ3)",
    "min(exp_co_occ3/base_co_occ3)",
    "max(exp_co_occ3)",
    "max(base_co_occ3)",
    "max(exp_co_occ3/base_co_occ3)",
    # co_occ3 weighted
    "avg(exp_co_occ3_w)",
    "avg(base_co_occ3_w)",
    "avg(exp_co_occ3_w/base_co_occ3_w))",
    "min(exp_co_occ3_w)",
    "min(base_co_occ3_w)",
    "min(exp_co_occ3_w/base_co_occ3_w)",
    "max(exp_co_occ3_w)",
    "max(base_co_occ3_w)",
    "max(exp_co_occ3_w/base_co_occ3_w)",
)

assert len(_feature_names) == _num_features
