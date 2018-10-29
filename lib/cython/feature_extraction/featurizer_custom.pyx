from featurizer_collection_stats import FeaturizerCollectionStats
import numpy as np
from numpy cimport ndarray
cimport numpy as np
cimport cython

from libc.math cimport log2, sqrt


class FeaturizerCustom(FeaturizerCollectionStats):
    def __init__(self, collection_stats, collection_stats_segment_to_segment_id, *args, **kwargs):
        super(FeaturizerCustom, self).__init__(
            feature_names=_num_features,
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
    size_t * max_co_occ2,
    size_t * max_co_occ2_weighted,
    size_t * max_co_occ3,
    float * avg_co_occ2,
    float * avg_co_occ2_weighted,
    float * avg_co_occ3
):
    # temporary variables
    cdef size_t andpos1, andpos2
    cdef size_t best_occ2_freq, best_occ2_gap, best_occ3_freq
    cdef size_t occ_2_freq, occ_2_min_gap, occ_3_freq
    cdef size_t avg_co_occ2_denominator = 0, avg_co_occ3_denominator = 0

    # reset the pointed values
    max_co_occ2[0] = max_co_occ2_weighted[0] = max_co_occ3[0] = 0
    avg_co_occ2[0] = avg_co_occ2_weighted[0] = avg_co_occ3[0] = 0.0

    # co-occ2
    for andpos1, synset1 in enumerate(and_repr):
        if andpos1 == ref_andpos:
            continue
        best_occ2_freq = best_occ2_gap = 0
        avg_co_occ2_denominator += len(synset1)
        for termid_term_tags in synset1:
            occ_2_freq, occ_2_min_gap = collection_stats.get_co_occ2_freq_min_gap_tuple(termid_term_tags[0], ref_termid)
            if occ_2_freq > best_occ2_freq or (occ_2_freq == best_occ2_freq and occ_2_min_gap < best_occ2_gap):
                best_occ2_freq = occ_2_freq
                best_occ2_gap = occ_2_min_gap
            avg_co_occ2[0] += occ_2_freq
            avg_co_occ2_weighted[0] += occ_2_freq * occ_2_min_gap

        max_co_occ2[0] += best_occ2_freq
        max_co_occ2_weighted[0] += best_occ2_freq * best_occ2_gap

    if avg_co_occ2_denominator > 0:
        avg_co_occ2[0] /= avg_co_occ2_denominator
    if avg_co_occ2_denominator > 0:
        avg_co_occ2_weighted[0] /= avg_co_occ2_denominator

    if max_co_occ2[0] == 0:
        return

    for andpos1, synset1 in enumerate(and_repr):
        if andpos1 == ref_andpos:
            continue
        for andpos2, synset2 in enumerate(and_repr):
            if andpos2 <= andpos1 or andpos2 == ref_andpos:
                continue
            best_occ3_freq = 0
            avg_co_occ3_denominator += len(synset1) * len(synset2)
            for termid_term_tags1 in synset1:
                for termid_term_tags2 in synset2:
                    occ_3_freq = collection_stats.get_co_occ3_freq(ref_termid, termid_term_tags1[0], termid_term_tags2[0])
                    if occ_3_freq > best_occ3_freq:
                        best_occ3_freq = occ_3_freq
                    avg_co_occ3[0] += occ_3_freq
            max_co_occ3[0] += best_occ3_freq
    if avg_co_occ3_denominator > 0:
        avg_co_occ3[0] /= avg_co_occ3_denominator
    # END

cdef float min_float32 = 1e-38


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
    collection_stats
):
    cdef np.float32_t* it_features

    # temporary variables
    cdef list base_and_query
    cdef size_t n, total_num_terms
    cdef size_t min_tf, min_df
    cdef size_t max_tf, max_df
    cdef size_t sum_tf, sum_df
    cdef size_t sum_tf_square, sum_df_square

    cdef size_t i_and, i_term
    cdef size_t and_query_pos
    cdef size_t starting_id
    cdef size_t exp_id

    cdef size_t and_query_size
    cdef size_t and_query_size_minus_one

    cdef size_t starting_tf, starting_df
    max_and_length = max(len(and_query) for and_query in exp_repr)
    cdef ndarray[np.uint64_t, ndim=1] base_num_terms = np.empty(max_and_length, dtype=np.uint64)
    cdef ndarray[np.uint64_t, ndim=1] base_min_tfs = np.empty(max_and_length, dtype=np.uint64)
    cdef ndarray[np.uint64_t, ndim=1] base_min_dfs = np.empty(max_and_length, dtype=np.uint64)
    cdef ndarray[np.uint64_t, ndim=1] base_max_tfs = np.empty(max_and_length, dtype=np.uint64)
    cdef ndarray[np.uint64_t, ndim=1] base_max_dfs = np.empty(max_and_length, dtype=np.uint64)
    cdef ndarray[np.float32_t, ndim=1] base_avg_tfs = np.empty(max_and_length, dtype=np.float32)
    cdef ndarray[np.float32_t, ndim=1] base_avg_dfs = np.empty(max_and_length, dtype=np.float32)
    cdef ndarray[np.float32_t, ndim=1] base_std_tfs = np.empty(max_and_length, dtype=np.float32)
    cdef ndarray[np.float32_t, ndim=1] base_std_dfs = np.empty(max_and_length, dtype=np.float32)
    cdef size_t base_max_num_terms
    cdef float base_avg_num_terms
    cdef size_t base_min_tf, base_min_df, base_max_tf, base_max_df
    cdef float base_avg_tf, base_avg_df, base_std_tf, base_std_df

    cdef size_t exp_tf = 0, exp_df = 0
    cdef size_t sum_max_tfs = 0, sum_max_dfs = 0

    cdef size_t starting_max_co_occ2, starting_max_co_occ2_weighted, starting_max_co_occ3
    cdef float  starting_avg_co_occ2, starting_avg_co_occ2_weighted, starting_avg_co_occ3
    cdef float  starting_avg_max_co_occ2, starting_avg_max_co_occ2_weighted, starting_avg_max_co_occ3
    #cdef float  starting_std_co_occ2, starting_std_co_occ2_weighted, starting_std_co_occ3

    cdef size_t base_max_co_occ2, base_max_co_occ2_weighted, base_max_co_occ3
    cdef float  base_avg_max_co_occ2, base_avg_max_co_occ2_weighted, base_avg_max_co_occ3
    cdef float  base_avg_co_occ2, base_avg_co_occ2_weighted, base_avg_co_occ3
    #cdef float  base_std_co_occ2, base_std_co_occ2_weighted, base_std_co_occ3
    #cdef float  base_std_max_co_occ2, base_std_max_co_occ2_weighted, base_std_max_co_occ3

    cdef size_t exp_max_co_occ2, exp_max_co_occ2_weighted, exp_max_co_occ3
    cdef float exp_avg_max_co_occ2, exp_avg_max_co_occ2_weighted, exp_avg_max_co_occ3
    cdef float  exp_avg_co_occ2, exp_avg_co_occ2_weighted, exp_avg_co_occ3
    #cdef float  exp_std_co_occ2, exp_std_co_occ2_weighted, exp_std_co_occ3

    # loop over the expansions
    cdef int it = from_row
    for i_and, and_query in enumerate(exp_repr):  # the query is the OR composition of different AND_QUERIES
        base_and_query = base_repr[i_and]

        and_query_size = len(and_query)
        and_query_size_minus_one = and_query_size - 1

        # normalization factors
        base_min_tf = base_min_df = 0
        base_max_tf = base_max_df = 0
        base_avg_tf = base_avg_df = 0
        base_std_tf = base_std_df = 0

        sum_max_tfs = sum_max_dfs = 0
        total_num_terms = 0
        base_avg_num_terms = base_max_num_terms = 0
        for and_query_pos, base_synset in enumerate(base_and_query):
            min_tf = min_df = 0
            max_tf = max_df = 0
            sum_tf = sum_df = 0
            sum_tf_square = sum_df_square = 0
            for base_termid_term_tags in base_synset:
                tpl = collection_stats.get_tf_df_tuple(base_termid_term_tags[0])
                sum_tf += tpl[0]
                sum_df += tpl[1]
                sum_tf_square += tpl[0] ** 2
                sum_df_square += tpl[1] ** 2
                if tpl[1] < min_df or (tpl[1] == min_df and tpl[0] < min_tf) or (min_df == 0):
                    min_df = tpl[1]
                    min_tf = tpl[0]
                if tpl[1] > max_df or (tpl[1] == max_df and tpl[0] > max_tf):
                    max_df = tpl[1]
                    max_tf = tpl[0]

            n = len(base_synset)
            total_num_terms += n

            # updata base properties related to the synset
            base_num_terms[and_query_pos] = n
            base_min_tfs[and_query_pos] = min_tf
            base_min_dfs[and_query_pos] = min_df
            base_max_tfs[and_query_pos] = max_tf
            base_max_dfs[and_query_pos] = max_df
            base_avg_tfs[and_query_pos] = 1.0 * sum_tf / n
            base_avg_dfs[and_query_pos] = 1.0 * sum_df / n
            base_std_tfs[and_query_pos] = sqrt(1.0 * sum_tf_square / n - 1.0 * sum_tf ** 2 / (n ** 2))
            base_std_dfs[and_query_pos] = sqrt(1.0 * sum_df_square / n - 1.0 * sum_df ** 2 / (n ** 2))

            sum_max_tfs += max_tf
            sum_max_dfs += max_df

            # update base properties related to the scope
            if n > base_max_num_terms:
                base_max_num_terms = n
            base_avg_num_terms += n
            if min_df < base_min_df or (min_df == base_min_df and min_tf < base_min_tf) or (base_min_df == 0):
                base_min_tf = min_tf
                base_min_df = min_df
            if max_df > base_max_df or (max_df == base_max_df and max_tf > base_max_tf):
                base_max_tf = max_tf
                base_max_df = max_df
            base_avg_tf += sum_tf
            base_avg_df += sum_df
            base_std_tf += sum_tf_square
            base_std_df += sum_df_square
        # in the following base_avg_num_terms is the sum of the number of terms, base_avg_tf is the tf sum and base_std_tf is the squared sum of the tf
        base_std_tf = sqrt(1.0 * base_std_tf / base_avg_num_terms - 1.0 * base_avg_tf ** 2 / (base_avg_num_terms ** 2))
        base_std_df = sqrt(1.0 * base_std_df / base_avg_num_terms - 1.0 * base_avg_df ** 2 / (base_avg_num_terms ** 2))
        # adjust the averages dividing for the right denominator
        base_avg_num_terms = 1.0 * base_avg_num_terms / and_query_size if and_query_size else 0.0
        base_avg_tf = 1.0 * base_avg_tf / total_num_terms if total_num_terms else 0.0
        base_avg_df = 1.0 * base_avg_df / total_num_terms if total_num_terms else 0.0

        # loop over the synsets (the OR groups)
        for and_query_pos, synset in enumerate(and_query):  # the AND_QUERY is the AND composition of different SYNSET (CNF)
            # base co-occurrences + starting co-occurrences (the last one is computed for free iterating in reverse order)
            starting_id = -1
            base_max_co_occ2 = base_max_co_occ2_weighted = base_max_co_occ3 = 0
            base_avg_max_co_occ2 = base_avg_max_co_occ2_weighted = base_avg_max_co_occ3 = 0
            base_avg_co_occ2 = base_avg_co_occ2_weighted = base_avg_co_occ3 = 0
            for i_term in range(len(base_and_query[and_query_pos]), 0, -1):  # the reverse order must be preserved
                starting_id = base_and_query[and_query_pos][i_term-1][0]
                _c_get_cooccurrences(
                    and_query_pos, starting_id, base_and_query, collection_stats,
                    &starting_max_co_occ2, &starting_max_co_occ2_weighted, &starting_max_co_occ3,
                    &starting_avg_co_occ2, &starting_avg_co_occ2_weighted, &starting_avg_co_occ3,
                )

                if starting_max_co_occ2 > base_max_co_occ2 or (starting_max_co_occ2 == base_max_co_occ2 and base_max_co_occ2_weighted > starting_max_co_occ2_weighted):
                    base_max_co_occ2 = starting_max_co_occ2
                    base_max_co_occ2_weighted = starting_max_co_occ2_weighted
                if starting_max_co_occ3 > base_max_co_occ3:
                    base_max_co_occ3 = starting_max_co_occ3
                base_avg_max_co_occ2 += starting_max_co_occ2
                base_avg_max_co_occ2_weighted += starting_max_co_occ2_weighted
                base_avg_max_co_occ3 += starting_max_co_occ3
                base_avg_co_occ2 += starting_avg_co_occ2
                base_avg_co_occ2_weighted += starting_avg_co_occ2_weighted
                base_avg_co_occ3 += starting_avg_co_occ3

            # the last element (in reverse order) is the starting id
            starting_tf, starting_df = collection_stats.get_tf_df_tuple(starting_id)

            n = len(base_and_query[and_query_pos])
            starting_avg_max_co_occ2 = (1.0 * starting_max_co_occ2 / and_query_size_minus_one) if and_query_size_minus_one else 0.0
            starting_avg_max_co_occ2_weighted = (1.0 * starting_max_co_occ2_weighted / and_query_size_minus_one) if and_query_size_minus_one else 0.0
            starting_avg_max_co_occ3 = (1.0 * starting_max_co_occ2 / and_query_size_minus_one) if and_query_size_minus_one else 0.0
            base_avg_max_co_occ2 /= (n * and_query_size_minus_one) if and_query_size_minus_one else 1.0
            base_avg_max_co_occ2_weighted /= (n * and_query_size_minus_one) if and_query_size_minus_one else 1.0
            base_avg_max_co_occ3 /= (n * and_query_size_minus_one) if and_query_size_minus_one else 1.0
            base_avg_co_occ2 /= n
            base_avg_co_occ2_weighted /= n
            base_avg_co_occ3 /= n

            for termid_term_tags in synset:  # the SYNSET is the OR composition of different synonyms
                it_features = &global_features[it,from_column]
                it += 1

                # exp term id and term frequency
                exp_id = termid_term_tags[0]
                exp_tf, exp_df = collection_stats.get_tf_df_tuple(exp_id)

                # exp co-occurrences
                _c_get_cooccurrences(
                    and_query_pos, exp_id, base_and_query, collection_stats,
                    &exp_max_co_occ2, &exp_max_co_occ2_weighted, &exp_max_co_occ3,
                    &exp_avg_co_occ2, &exp_avg_co_occ2_weighted, &exp_avg_co_occ3
                )

                exp_avg_max_co_occ2 = (1.0 * exp_max_co_occ2 / and_query_size_minus_one) if and_query_size_minus_one else 0.0
                exp_avg_max_co_occ2_weighted = (1.0 * exp_max_co_occ2_weighted / and_query_size_minus_one) if and_query_size_minus_one else 0.0
                exp_avg_max_co_occ3 = (1.0 * exp_max_co_occ3 / and_query_size_minus_one) if and_query_size_minus_one else 0.0

                # global statistics depending only from the scope
                it_features[ 0] = base_avg_num_terms
                it_features[ 1] = base_max_num_terms
                it_features[ 2] = base_min_tf
                it_features[ 3] = base_min_df
                it_features[ 4] = base_max_tf
                it_features[ 5] = base_max_df
                it_features[ 6] = base_avg_tf
                it_features[ 7] = base_avg_df
                it_features[ 8] = base_std_tf
                it_features[ 9] = base_std_df

                # global statistics depending only from the starting synset: TFs and DFs
                it_features[10] = starting_tf
                it_features[11] = starting_df

                it_features[12] = base_min_tfs[and_query_pos]
                it_features[13] = base_min_dfs[and_query_pos]

                it_features[14] = base_max_tfs[and_query_pos]
                it_features[15] = base_max_dfs[and_query_pos]

                it_features[16] = base_avg_tfs[and_query_pos]
                it_features[17] = base_avg_dfs[and_query_pos]

                it_features[18] = base_std_tfs[and_query_pos]
                it_features[19] = base_std_dfs[and_query_pos]

                # global features depending only from the starting synset: CO-OCCURRENCEs
                it_features[20] = starting_max_co_occ2
                it_features[21] = starting_max_co_occ2_weighted
                it_features[22] = starting_max_co_occ3

                it_features[23] = starting_avg_max_co_occ2
                it_features[24] = starting_avg_max_co_occ2_weighted
                it_features[25] = starting_avg_max_co_occ3

                it_features[26] = starting_avg_co_occ2
                it_features[27] = starting_avg_co_occ2_weighted
                it_features[28] = starting_avg_co_occ3

                it_features[29] = base_max_co_occ2
                it_features[30] = base_max_co_occ2_weighted
                it_features[31] = base_max_co_occ3

                it_features[32] = base_avg_max_co_occ2
                it_features[33] = base_avg_max_co_occ2_weighted
                it_features[34] = base_avg_max_co_occ3

                it_features[35] = base_avg_co_occ2
                it_features[36] = base_avg_co_occ2_weighted
                it_features[37] = base_avg_co_occ3

                # basic exp statistics: TFs and DFs
                it_features[38] = exp_tf
                it_features[39] = exp_df

                it_features[40] = (1.0 * exp_tf / starting_tf) if starting_tf > min_float32 else 0.0
                it_features[41] = (1.0 * exp_df / starting_df) if starting_tf > min_float32 else 0.0

                it_features[42] = (1.0 * exp_tf * and_query_size / sum_max_tfs) if sum_max_tfs > min_float32 else 0.0
                it_features[43] = (1.0 * exp_df * and_query_size / sum_max_dfs) if sum_max_dfs > min_float32 else 0.0

                # other exp statistics: CO-OCCURRENCEs
                if exp_max_co_occ2 > 0:
                    it_features[44] = exp_max_co_occ2
                    it_features[45] = exp_max_co_occ2_weighted
                    it_features[46] = exp_max_co_occ3

                    it_features[47] = (1.0 * exp_max_co_occ2 / starting_max_co_occ2) if starting_max_co_occ2 > min_float32 else 0.0
                    it_features[48] = (1.0 * exp_max_co_occ2_weighted / starting_max_co_occ2_weighted) if starting_max_co_occ2_weighted > min_float32 else 0.0
                    it_features[49] = (1.0 * exp_max_co_occ3 / starting_max_co_occ3) if starting_max_co_occ3 > min_float32 else 0.0

                    it_features[50] = (1.0 * exp_max_co_occ2 / base_max_co_occ2) if base_max_co_occ2 > min_float32 else 0.0
                    it_features[51] = (1.0 * exp_max_co_occ2_weighted / base_max_co_occ2_weighted) if base_max_co_occ2_weighted > min_float32 else 0.0
                    it_features[52] = (1.0 * exp_max_co_occ3 / base_max_co_occ3) if base_max_co_occ3 > min_float32 else 0.0

                    it_features[53] = exp_avg_max_co_occ2
                    it_features[54] = exp_avg_max_co_occ2_weighted
                    it_features[55] = exp_avg_max_co_occ3

                    it_features[56] = (1.0 * exp_avg_max_co_occ2 / starting_avg_max_co_occ2) if starting_avg_max_co_occ2 > min_float32 else 0.0
                    it_features[57] = (1.0 * exp_avg_max_co_occ2_weighted / starting_avg_max_co_occ2_weighted) if starting_avg_max_co_occ2_weighted > min_float32 else 0.0
                    it_features[58] = (1.0 * exp_avg_max_co_occ3 / starting_avg_max_co_occ3) if starting_avg_max_co_occ3 > min_float32 else 0.0

                    it_features[59] = (1.0 * exp_avg_max_co_occ2 / base_avg_max_co_occ2) if base_avg_max_co_occ2 > min_float32 else 0.0
                    it_features[60] = (1.0 * exp_avg_max_co_occ2_weighted / base_avg_max_co_occ2_weighted) if base_avg_max_co_occ2_weighted > min_float32 else 0.0
                    it_features[61] = (1.0 * exp_avg_max_co_occ3 / base_avg_max_co_occ3) if base_avg_max_co_occ3 > min_float32 else 0.0

                    it_features[62] = exp_avg_co_occ2
                    it_features[63] = exp_avg_co_occ2_weighted
                    it_features[64] = exp_avg_co_occ3

                    it_features[65] = (1.0 * exp_avg_co_occ2 / starting_avg_co_occ2) if starting_avg_co_occ2 > min_float32 else 0.0
                    it_features[66] = (1.0 * exp_avg_co_occ2_weighted / starting_avg_co_occ2_weighted) if starting_avg_co_occ2_weighted > min_float32 else 0.0
                    it_features[67] = (1.0 * exp_avg_co_occ3 / starting_avg_co_occ3) if starting_avg_co_occ3 > min_float32 else 0.0

                    it_features[68] = (1.0 * exp_avg_co_occ2 / base_avg_co_occ2) if base_avg_co_occ2 > min_float32 else 0.0
                    it_features[69] = (1.0 * exp_avg_co_occ2_weighted / base_avg_co_occ2_weighted) if base_avg_co_occ2_weighted > min_float32 else 0.0
                    it_features[70] = (1.0 * exp_avg_co_occ3 / base_avg_co_occ3) if base_avg_co_occ3 > min_float32 else 0.0
                else:
                    for f in range(44, 71):
                        it_features[f] = 0


cdef int _num_features = 71
