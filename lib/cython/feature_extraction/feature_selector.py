from featurizer import Featurizer

class FeatureSelector(Featurizer):
    def __init__(self, featurizer, feature_position_list, *args, **kwargs):
        assert isinstance(featurizer, Featurizer)
        assert hasattr(feature_position_list, "__iter__") and all(isinstance(feature_position, int) for feature_position in feature_position_list)
        assert min(feature_position_list) >= 0 and max(feature_position_list) < featurizer.num_features

        feature_names = [featurizer.feature_name(feature_position) for feature_position in feature_position_list]
        super(FeatureComposer, self).__init__(feature_names, *args, **kwargs)
        self._featurizer = featurizer
        self._feature_position_list = tuple(feature_position_list)

    def _transform_impl(
            self,
            base_repr, exp_repr, num_exp_terms,
            global_features, from_row, from_column
    ):
        raw_features = self._featurizer.transform(base_repr, exp_repr, num_exp_terms)
        to_row = from_row + num_exp_terms
        to_column = from_column + len(feature_position_list)

        global_features[from_row:to_row, from_column:to_column] = raw_features[:,self._feature_position_list]

