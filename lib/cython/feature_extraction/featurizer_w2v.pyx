from featurizer import Featurizer

from gensim import matutils
from gensim.models import Word2Vec
from gensim.models.keyedvectors import Vocab

import numpy as np
from numpy cimport ndarray
cimport numpy as np
cimport cython


class FeaturizerW2V(Featurizer):
    def __init__(self, word2vec, *args, **kwargs):
        assert isinstance(word2vec, Word2Vec)
        assert word2vec.negative, "We have currently only implemented predict_output_word for the negative sampling scheme"
        super(FeaturizerW2V, self).__init__(feature_names=_feature_names, *args, **kwargs)
        self._word2vec = word2vec

    def _transform_impl(
            self,
            base_repr, exp_repr, num_exp_terms,
            global_features, from_row, from_column
    ):
        # convert the two representations using the word2vec vocabulary
        v = self._word2vec.wv.vocab
        none_vocab = Vocab(index=None)

        base_repr = [
            [
                [
                    v.get("_{}_".format(syn_tags[0].replace(" ", "_")) if " " in syn_tags[0] else syn_tags[0], none_vocab).index
                    for syn_tags in synset
                ]
                for synset in and_query
            ]
            for and_query in base_repr
        ]
        exp_repr = [
            [
                [
                    v.get("_{}_".format(syn_tags[0].replace(" ", "_")) if " " in syn_tags[0] else syn_tags[0], none_vocab).index
                    for syn_tags in synset
                ]
                for synset in and_query
            ]
            for and_query in exp_repr
        ]

        _c_get_features(
            base_repr, exp_repr, num_exp_terms,
            global_features, from_row, from_column,
            self._word2vec
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
        object word2vec
):
    cdef np.float32_t* it_features

    v_syn0 = word2vec.wv.syn0
    v_syn0norm = word2vec.wv.syn0norm
    v_syn1 = word2vec.syn1neg

    # temporary variables
    cdef size_t i_and = 0, and_query_pos = 0
    cdef ndarray[np.float32_t, ndim=2] base_and_avg_syn0, base_and_avg_syn1
    #cdef ndarray[np.float32_t, ndim=1] base_and_avg_syn0_sum, base_and_avg_syn0_context
    cdef size_t context_vectors
    cdef ndarray[np.uint32_t, ndim=1] base_and_size
    cdef ndarray[np.double_t, ndim=1] base_term_avg_syn0norm, base_term_syn0norm
    cdef ndarray[np.double_t, ndim=1] base_term_avg_syn1norm, base_term_syn1norm
    cdef size_t synset_size, base_context_size

    # loop over the expansions
    cdef int it = from_row
    cdef double max_syn0_sim, avg_syn0_sim, max_syn1_sim, avg_syn1_sim, prob_value
    for i_and, and_query in enumerate(exp_repr):  # the query is the OR composition of different AND_QUERIES

        # init the main query representations
        context_vectors = 0
        base_and_size = np.empty(len(and_query), np.uint32)
        base_and_avg_syn0 = np.empty((len(and_query), word2vec.vector_size), np.float32)
        base_and_avg_syn1 = np.empty((len(and_query), word2vec.vector_size), np.float32)
        for and_query_pos, base_synset in enumerate(base_repr[i_and]):
            base_syn0 = [v_syn0[base_index] for base_index in base_synset if base_index is not None]
            base_syn1 = [v_syn1[base_index] for base_index in base_synset if base_index is not None]
            synset_size = len(base_syn0)

            base_and_size[and_query_pos] = synset_size
            if synset_size == 0:
                base_and_avg_syn0[and_query_pos,:] = 0.0
                base_and_avg_syn1[and_query_pos,:] = 0.0
            else:
                # compute the average of each synset
                np.sum(
                    base_syn0,
                    axis=0,
                    out=base_and_avg_syn0[and_query_pos,:],
                    dtype=np.float
                )
                np.sum(
                    base_syn1,
                    axis=0,
                    out=base_and_avg_syn1[and_query_pos,:],
                    dtype=np.float
                )
                base_and_avg_syn0[and_query_pos,:] /= synset_size
                base_and_avg_syn1[and_query_pos,:] /= synset_size
                context_vectors += 1
        base_and_avg_syn0_sum = base_and_avg_syn0.sum(axis=0)
        # init end

        for and_query_pos, synset in enumerate(and_query):  # the AND_QUERY is the AND composition of different SYNSET (CNF)
            synset_size = len(synset)
            # DON'T MODIFY the elements of base_syn0norms
            base_syn0norms = [v_syn0norm[base_index] for base_index in base_repr[i_and][and_query_pos] if base_index is not None]
            base_syn1norms = [matutils.unitvec(v_syn1[base_index]) for base_index in base_repr[i_and][and_query_pos] if base_index is not None]
            base_term_avg_syn0norm = matutils.unitvec(base_and_avg_syn0[and_query_pos,:])
            base_term_avg_syn1norm = matutils.unitvec(base_and_avg_syn1[and_query_pos,:])

            base_context_size = context_vectors - (base_and_size[and_query_pos] > 0)
            base_and_avg_syn0_context = base_and_avg_syn0_sum - base_and_avg_syn0[and_query_pos,:]
            if base_context_size > 0 and word2vec.cbow_mean:
                base_and_avg_syn0_context /= base_context_size

            for term_index in synset:  # the SYNSET is the OR composition of different synonyms
                it_features = &global_features[it,from_column]
                it += 1

                max_syn0_sim = max_syn1_sim = 0.0
                avg_syn0_sim = avg_syn1_sim = 0.0
                prob_value = 0.0
                if term_index is not None:
                    if base_and_size[and_query_pos] > 0:
                        # DON'T MODIFY term_repr
                        term_syn0norm = v_syn0norm[term_index]
                        term_syn1norm = matutils.unitvec(v_syn1[term_index])

                        max_syn0_sim = max(
                            np.dot(term_syn0norm, base_term_syn0norm)
                            for base_term_syn0norm in base_syn0norms
                        )
                        max_syn1_sim = max(
                            np.dot(term_syn1norm, base_term_syn1norm)
                            for base_term_syn1norm in base_syn1norms
                        )
                        # double check when only one term is in the base synset
                        # assert word2vec.similarity(word2vec.wv.index2word[term_index], word2vec.wv.index2word[base_repr[i_and][and_query_pos][0]]) == max_syn0_sim

                        avg_syn0_sim = np.dot(term_syn0norm, base_term_avg_syn0norm)
                        avg_syn1_sim = np.dot(term_syn1norm, base_term_avg_syn1norm)
                    if base_context_size > 0:
                        prob_value = np.exp(np.dot(base_and_avg_syn0_context, v_syn1[term_index]))

                it_features[0] = max_syn0_sim
                it_features[1] = avg_syn0_sim
                it_features[2] = max_syn1_sim
                it_features[3] = avg_syn1_sim
                it_features[4] = prob_value
            # normalize the probabilities
            if synset_size:
                global_features[it-synset_size:it,from_column+4] /= max(np.sum(global_features[it-synset_size:it,from_column+4]), 1.0)

# this double check avoid mistakes
cdef int _num_features = 5
cdef tuple _feature_names = (
    "max_syn0_sim",
    "avg_syn0_sim",
    "max_syn1_sim",
    "avg_syn1_sim",
    "context_prob"
)

assert len(_feature_names) == _num_features
