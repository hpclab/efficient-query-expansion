from featurizer_collection_stats import FeaturizerCollectionStats
import numpy as np
from numpy cimport ndarray
cimport numpy as np
cimport cython

from libc.math cimport log2, sqrt


class FeaturizerQueryPerformancePredictors(FeaturizerCollectionStats):
    def __init__(self, collection_stats, collection_stats_segment_to_segment_id, *args, **kwargs):
        super(FeaturizerQueryPerformancePredictors, self).__init__(
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
cdef float _c_variability(
    size_t values_sum,
    size_t squared_values_sum,
    int num_values
):
    if num_values <= 1 or values_sum == 0:
        return 0
    return (squared_values_sum - 1.0 * (values_sum ** 2) / num_values) / num_values


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef float _c_standard_deviation(
    float values_sum,
    float squared_values_sum,
    int num_values
):
    if num_values <= 1 or values_sum == 0:
        return 0
    return sqrt(
        (squared_values_sum - (values_sum ** 2) / num_values) / num_values
    )


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _c_fill_idf_tf_vector(
    list and_repr, float idf_multiplier, collection_stats,
    ndarray[np.float32_t, ndim=1] min_idf_vec,
    ndarray[np.uint64_t, ndim=1] max_df_vec,
    ndarray[np.uint64_t, ndim=1] max_tf_vec,
    ndarray[np.uint64_t, ndim=1] max_tf_square_vec,
):
    cdef size_t i_or
    cdef size_t tf, df
    cdef float idf

    for i_or, synset in enumerate(and_repr):
        max_df_vec[i_or] = max_tf_vec[i_or] = 0
        for termid_term_tags in synset:
            stats_term = collection_stats.get_stats_term(termid_term_tags[0])
            df = stats_term.df
            tf = stats_term.tf
            if df > max_df_vec[i_or] or (df == max_df_vec[i_or] and tf < max_tf_vec[i_or]):
                max_df_vec[i_or] = df
                max_tf_vec[i_or] = tf
                max_tf_square_vec[i_or] = stats_term.tf_square
        min_idf_vec[i_or] = idf_multiplier / (max_df_vec[i_or] + 1.0)


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

    # temporary variables
    cdef list base_and_query
    cdef size_t i_and, i_or
    cdef size_t exp_id

    cdef size_t num_base_terms
    cdef size_t num_and_terms, num_base_total_terms, num_syns
    cdef float base_idf, exp_idf
    cdef float exp_max_idf, exp_min_idf
    cdef size_t base_tf, base_df
    cdef size_t exp_tf, exp_df

    cdef size_t collection_num_docs = collection_stats.get_num_docs()
    cdef size_t collection_sum_term_frequency = collection_stats.get_term_frequency_sum()
    cdef float idf_multiplier = log2(collection_num_docs + 0.5) / log2(collection_num_docs + 1.0)
    cdef ndarray[np.float32_t, ndim=1] base_min_idf_vec = np.empty(max_len, dtype=np.float32)
    cdef ndarray[np.uint64_t, ndim=1] base_max_df_vec = np.empty(max_len, dtype=np.uint64)
    cdef ndarray[np.uint64_t, ndim=1] base_max_tf_vec = np.empty(max_len, dtype=np.uint64)
    cdef ndarray[np.uint64_t, ndim=1] base_max_tf_square_vec = np.empty(max_len, dtype=np.uint64)
    cdef float sum_idf, sum_squared_idf
    cdef float base_std_dev_idf, exp_std_dev_idf
    cdef float base_min_idf, base_max_idf
    cdef float sum_ictf, exp_av_ictf
    cdef float sum_qcs, max_qcs, base_cs, exp_cs, exp_qcs, exp_max_qcs
    cdef float sum_qvar, max_qvar, base_qvar, exp_qvar, exp_sum_qvar, exp_max_qvar

    # loop over the expansions
    cdef int it = from_row
    for i_and, and_query in enumerate(exp_repr):  # the query is the OR composition of different AND_QUERIES
        base_and_query = base_repr[i_and]
        num_and_terms = len(base_and_query)
        num_base_total_terms = sum(len(synset) for synset in base_and_query)

        # vector with the minimum idf of each synset
        _c_fill_idf_tf_vector(base_and_query, idf_multiplier, collection_stats, base_min_idf_vec, base_max_df_vec, base_max_tf_vec, base_max_tf_square_vec)

        # set sum_idf and sum_squared_idf for the std_dev computation
        sum_idf = base_min_idf = base_max_idf = base_min_idf_vec[0]
        sum_squared_idf = base_min_idf_vec[0] ** 2
        for i_or in range(1, num_and_terms):
            base_idf = base_min_idf_vec[i_or]
            sum_idf += base_idf
            sum_squared_idf += base_idf ** 2
            if base_idf >= base_max_idf:
                base_max_idf = base_idf
            elif base_idf < base_min_idf:
                base_min_idf = base_idf
        base_std_dev_idf = _c_standard_deviation(sum_idf, sum_squared_idf, num_and_terms)

        # set sum_ictf
        sum_ictf = 0
        for i_or in range(num_and_terms):
            base_tf = base_max_tf_vec[i_or]
            sum_ictf += log2(collection_sum_term_frequency) - log2(base_tf)

        # set sum_qcs and max_qcs
        sum_qcs = max_qcs = 0
        for i_or in range(num_and_terms):
            base_cs = (1 + log2(base_max_tf_vec[i_or]+1)) / log2(1 + 1.0 * collection_num_docs / (base_max_df_vec[i_or]+1))
            sum_qcs += base_cs
            if base_cs > max_qcs:
                max_qcs = base_cs

        # set sum_qvar and max_qvar
        sum_qvar = max_qvar = 0
        for i_or in range(num_and_terms):
            base_qvar = _c_variability(base_max_tf_vec[i_or], base_max_tf_square_vec[i_or], base_max_df_vec[i_or])
            sum_qvar += base_qvar
            if base_qvar > max_qvar:
                max_qvar = base_qvar

        for i_or, synset in enumerate(and_query):  # the AND_QUERY is the AND composition of different SYNSET (CNF)
            num_base_terms = len(base_and_query[i_or])
            num_syns = len(synset)
            base_tf = base_max_tf_vec[i_or]
            base_df = base_max_df_vec[i_or]
            base_idf = base_min_idf_vec[i_or]
            base_cs = (1 + log2(base_tf+1)) / log2(1 + 1.0 * collection_num_docs / (base_df+1))
            base_qvar = _c_variability(base_tf, base_max_tf_square_vec[i_or], base_df)

            for termid_term_tags in synset:  # the SYNSET is the OR composition of different synonyms
                it_features = &global_features[it,from_column]
                it += 1

                # exp term id and term frequency
                exp_id = termid_term_tags[0]
                stats_exp = collection_stats.get_stats_term(exp_id)
                exp_df = stats_exp.df
                exp_tf = stats_exp.tf
                exp_idf = idf_multiplier / (exp_df + 1.0)
                exp_std_dev_idf = _c_standard_deviation(
                    sum_idf - base_idf + exp_idf,
                    sum_squared_idf - (base_idf ** 2) + (exp_idf ** 2),
                    num_and_terms
                )
                # compute the max idf among the expansions if this expansion term can affect the old maximum
                if base_idf <= exp_idf:
                    # the expansion doesn't change the base_min_idf vector
                    exp_max_idf = base_max_idf
                else:
                    if base_max_idf != base_idf:
                        # base_max_idf doesn't change after the swap in the vector
                        exp_max_idf = base_max_idf
                    else:
                        # the maximum must be recomputed
                        base_min_idf_vec[i_or] = exp_idf
                        exp_max_idf = base_min_idf_vec[:num_and_terms].max()
                        base_min_idf_vec[i_or] = base_idf
                exp_min_idf = min(base_min_idf, exp_idf)

                # compute max ictf among the expansions
                if base_tf >= exp_tf:
                    exp_av_ictf = sum_ictf / num_and_terms
                else:
                    # since - log2(base_tf) appears with both signs I can simplify the formula
                    exp_av_ictf = (sum_ictf -(-log2(base_tf)) +(-log2(exp_tf))) / num_and_terms

                # compute qcs and max_qcs
                exp_cs = (1 + log2(exp_tf+1)) / log2(1 + 1.0 * collection_num_docs / (exp_df+1))
                if base_cs >= exp_cs:
                    exp_qcs = sum_qcs
                    exp_max_qcs = max_qcs
                else:
                    exp_qcs = sum_qcs - base_cs + exp_cs
                    exp_max_qcs = max(max_qcs, exp_cs)

                # compute sum_qvar and max_qvar
                if exp_tf >= base_tf:
                    exp_qvar = _c_variability(exp_tf, stats_exp.tf_square, exp_df)
                    exp_sum_qvar = sum_qvar - base_qvar + exp_qvar
                    exp_max_qvar = exp_qvar if exp_qvar > max_qvar else max_qvar
                else:
                    exp_sum_qvar = sum_qvar
                    exp_max_qvar = max_qvar


                # general features based on the number of terms involved
                it_features[ 0] = num_and_terms
                it_features[ 1] = num_base_total_terms
                it_features[ 2] = num_base_total_terms - num_and_terms
                it_features[ 3] = num_base_terms
                it_features[ 4] = num_syns

                # idf based measures
                it_features[ 5] = base_idf
                it_features[ 6] = exp_idf
                it_features[ 7] = exp_idf / (base_idf or 1.0)

                it_features[ 8] = exp_std_dev_idf
                it_features[ 9] = exp_min_idf
                it_features[10] = exp_max_idf
                it_features[11] = exp_max_idf / (exp_min_idf or 1.0)

                # AvICTF Average Inverse Collection Term Frequency
                it_features[12] = exp_av_ictf

                # QCS and maxQCS
                it_features[13] = exp_qcs
                it_features[14] = exp_max_qcs

                # QVar and maxQVar
                it_features[15] = exp_sum_qvar
                it_features[16] = exp_max_qvar


# this double check avoid mistakes
cdef int _num_features = 17
cdef tuple _feature_names = (
    "num_and_components",
    "num_base_total_terms",
    "num_base_additional_terms",
    "num_base_terms",
    "num_syns",
    "base_idf",
    "idf",
    "idf/base_idf",
    "std_dev_idf",
    "min_idf",
    "max_idf",
    "max_idf/min_idf",
    "av_ictf",
    "qcs",
    "max_qcs",
    "qvar",
    "max_qvar",
)

assert len(_feature_names) == _num_features
