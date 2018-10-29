import cfg
import numpy
import os
import sys


if __name__ ==  "__main__":
    # create the support directories if needed
    if not os.path.exists(cfg.processed_dir):
        os.mkdir(cfg.processed_dir)
    if not os.path.exists(cfg.thesaurus_dir):
        os.mkdir(cfg.thesaurus_dir)
    if not os.path.exists(cfg.training_dir):
        os.mkdir(cfg.training_dir)
    if not os.path.exists(cfg.tmp_dir):
        os.mkdir(cfg.tmp_dir)

    # create init file in wikiextractor submodule
    with open(cfg.base_dir + "lib/python/wikiextractor/__init__.py") as f:
        pass

    # compile the cython extensions
    os.system("cd lib/cpp/pattern_matching/; mkdir build; python setup.py build_ext --build-lib ../; rm -r build")
    os.system("cd lib/cython/; mkdir build; python setup.py build_ext --inplace; rm -r build")
