from featurizer import Featurizer
import numpy as np


class FeatureNormalizer(Featurizer):
    def __init__(self, featurizer, normalization_name_function_list, *args, **kwargs):
        assert isinstance(featurizer, Featurizer)
        assert hasattr(normalization_name_function_list, "__iter__") and len(normalization_name_function_list) > 0
        assert all(isinstance(name, str) and hasattr(function, "__call__") for name, function in normalization_name_function_list)

        feature_names = []
        for normalization_name, normalization_function in normalization_name_function_list:
            for feature_name in featurizer.feature_names():
                feature_names.append("{}{}".format(normalization_name, feature_name))
        super(FeatureNormalizer, self).__init__(feature_names, *args, **kwargs)
        self._featurizer = featurizer
        self._normalization_name_function_list = tuple((name, function) for name, function in normalization_name_function_list)

    def _transform_impl(
            self,
            base_repr, exp_repr, num_exp_terms,
            global_features, from_row, from_column
    ):
        num_features = self._featurizer.num_features
        raw_features = self._featurizer.transform(base_repr, exp_repr, num_exp_terms)
        assert raw_features.max() is not np.nan
        if num_exp_terms == 0 or num_features == 0:
            return
        for normalization_name, normalization_function in self._normalization_name_function_list:
            global_features[from_row:(from_row+num_exp_terms),from_column:(from_column+num_features)] = normalization_function(raw_features)
            from_column += num_features
