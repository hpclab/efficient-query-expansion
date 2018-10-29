import numpy
import os
import sys
from Cython.Build import cythonize
from distutils.core import setup, Extension


def add_extension(extensions, name, **kwargs):
    assert isinstance(name, str) and "/" not in name
    assert name not in extensions

    source = name.replace(".", "/") + ".pyx"
    if "language" not in kwargs:
        kwargs["language"] = "c++"
    if "extra_compile_args" in kwargs:
        kwargs["extra_compile_args"] += ["-std=c++11", "-O3"]
    else:
        kwargs["extra_compile_args"] = ["-std=c++11", "-O3"]

    extensions[name] = Extension(
        name,
        sources=[source],
        **kwargs
    )


if __name__ ==  "__main__":
    sys.path.append("../../")
    import cfg
    extensions = e = dict()

    # set the compiler
    os.environ["CC"] = "g++-7"

    # collection_stats
    kwargs = {"include_dirs": []}
    kwargs["include_dirs"].append(cfg.lib_dir + "cpp")
    kwargs["include_dirs"].append(cfg.lib_dir + "cpp/pattern_matching")
    add_extension(e, 'collection_stats.collection_stats', extra_link_args=['-fopenmp'], extra_compile_args=['-fopenmp'], **kwargs)
    add_extension(e, 'collection_stats.collection_stats_restricted', extra_link_args=['-fopenmp'], extra_compile_args=['-fopenmp'], **kwargs)
    # featurizers
    kwargs["include_dirs"].append(numpy.get_include())
    add_extension(e, 'feature_extraction.featurizer_textual', **kwargs)
    add_extension(e, 'feature_extraction.featurizer_tags', **kwargs)
    add_extension(e, 'feature_extraction.featurizer_w2v', **kwargs)
    add_extension(e, 'feature_extraction.featurizer_sigir08', **kwargs)
    add_extension(e, 'feature_extraction.featurizer_sigir08extended', **kwargs)
    add_extension(e, 'feature_extraction.featurizer_custom', **kwargs)
    add_extension(e, 'feature_extraction.featurizer_qpp', **kwargs)

    # setup
    setup(
        ext_modules=cythonize(extensions.values()),
        packages=extensions.keys(),
    )
