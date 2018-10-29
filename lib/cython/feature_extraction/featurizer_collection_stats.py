from featurizer import Featurizer
import collection_stats.collection_stats as cs
import collection_stats.collection_stats_restricted as csr

class FeaturizerCollectionStats(Featurizer):
    def __init__(
            self,
            feature_names, collection_stats_feature_fun,
            collection_stats, collection_stats_segment_to_segment_id,
            *args, **kwargs
    ):
        assert hasattr(collection_stats_feature_fun, "__call__")
        assert isinstance(collection_stats, (cs.PyCollectionStats , csr.PyCollectionStatsRestricted))
        assert isinstance(collection_stats_segment_to_segment_id, dict) and all(isinstance(segment, str) and isinstance(segment_id, int) for segment, segment_id in collection_stats_segment_to_segment_id.iteritems())

        super(FeaturizerCollectionStats, self).__init__(feature_names, *args, **kwargs)
        self._collection_stats = collection_stats
        self._collection_stats_segment_to_segment_id = collection_stats_segment_to_segment_id
        self._collection_stats_feature_fun = collection_stats_feature_fun

    def _transform_impl(
            self,
            base_repr, exp_repr, num_exp_terms,
            global_features, from_row, from_column
    ):
        base_repr = [
            [[(self._collection_stats_segment_to_segment_id[syn_tag[0].strip()],) + syn_tag for syn_tag in synset]
             for synset in and_query] for and_query in base_repr
        ]
        exp_repr = [
            [[(self._collection_stats_segment_to_segment_id[syn_tag[0].strip()],) + syn_tag for syn_tag in synset]
             for synset in and_query] for and_query in exp_repr
        ]

        return self._collection_stats_feature_fun(
            base_repr, exp_repr, num_exp_terms,
            global_features, from_row, from_column,
            self._collection_stats
        )
