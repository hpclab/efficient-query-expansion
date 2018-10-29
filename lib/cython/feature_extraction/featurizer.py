import numpy as np

class Featurizer(object):
    def __init__(self, feature_names, feature_name_prefix="", feature_name_suffix=""):
        assert isinstance(feature_names, int) or (hasattr(feature_names, "__iter__") and all(isinstance(feature_name, str) for feature_name in feature_names))
        assert isinstance(feature_name_prefix, str)
        assert isinstance(feature_name_suffix, str)
        if isinstance(feature_names, int):
            feature_names = tuple("f{}".format(i) for i in xrange(feature_names))
        else:
            assert len(feature_names) == len(set(feature_names))

        self._feature_names = tuple(feature_name_prefix + feature_name + feature_name_suffix for feature_name in feature_names)
        self._feature_name_prefix = feature_name_prefix
        self._feature_name_suffix = feature_name_suffix
        self._num_features = len(self._feature_names)

    def _transform_impl(
            self,
            base_repr, exp_repr, num_exp_terms,
            global_features, from_row, from_column
    ):
        # this method must be overriden
        raise NotImplementedError()

    def transform(
            self,
            base_repr, exp_repr, num_exp_terms,
            global_features=None, from_row=0, from_column=0
    ):
        assert base_repr is not None
        assert exp_repr is not None
        assert num_exp_terms >= 0
        if global_features is None:
            assert from_row == 0 and from_column == 0
            global_features = np.empty((num_exp_terms, self._num_features), dtype=np.float32)
        else:
            assert from_row >= 0 and from_column >= 0
            assert from_row + num_exp_terms <= global_features.shape[0] and from_column + self._num_features <= global_features.shape[1]

        self._transform_impl(base_repr, exp_repr, num_exp_terms, global_features, from_row, from_column)

        return global_features

    def feature_name(self, i):
        return self._feature_names[i]

    def feature_name_position(self, feature_name):
        position = self._feature_names.index(feature_name)
        assert feature_position >= 0
        return feature_position

    def feature_names(self):
        return self._feature_names[:]

    def feature_names_positions(self, feature_names):
        feature_positions = [self._feature_names.index(feature_name) for feature_name in feature_names]
        assert all(feature_position >= 0 for feature_position in feature_positions)
        return feature_positions

    @property
    def feature_name_prefix(self):
        return self._feature_name_prefix

    @property
    def feature_name_suffix(self):
        return self._feature_name_suffix

    @property
    def num_features(self):
        return self._num_features

