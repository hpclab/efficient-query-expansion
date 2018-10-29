# CONFIG FILE
import os
import sys

# base directory
base_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
data_dir = base_dir + "data/"
lib_dir  = base_dir + "lib/"

# add the python libraries to the python path
sys.path.append(lib_dir + "python/")
sys.path.append(lib_dir + "cpp/")
sys.path.append(lib_dir + "cython/")

# other directories
processed_dir = data_dir + "processed/"
raw_dir       = data_dir + "raw/"
thesaurus_dir = data_dir + "thesaurus/"
training_dir  = data_dir + "training/"
tmp_dir       = data_dir + "tmp/"

# number of parts the wikipedia file must be splitted to
wiki_preprocessing_split_into = 10

# some checks for consistency
assert os.path.isdir(base_dir)
assert os.path.isdir(data_dir)
assert os.path.isdir(lib_dir)
