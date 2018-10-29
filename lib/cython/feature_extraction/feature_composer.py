from featurizer import Featurizer

class FeatureComposer(Featurizer):
    def __init__(self, featurizer_list, *args, **kwargs):
        assert hasattr(featurizer_list, "__iter__") and all(isinstance(featurizer, Featurizer) for featurizer in featurizer_list)

        feature_names = []
        for i, featurizer in enumerate(featurizer_list):
            for feature_name in featurizer.feature_names():
                feature_names.append("[{}]{}".format(i, feature_name))
        super(FeatureComposer, self).__init__(feature_names, *args, **kwargs)
        self._featurizer_list = tuple(featurizer_list)

    def _transform_impl(
            self,
            base_repr, exp_repr, num_exp_terms,
            global_features, from_row, from_column
    ):
        delta = 0
        for featurizer in self._featurizer_list:
            featurizer.transform(base_repr, exp_repr, num_exp_terms, global_features, from_row, from_column+delta)
            delta += featurizer.num_features
