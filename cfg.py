# CONFIG FILE
import os
import sys

# base directory
base_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
# add the python libraries to the sys path
sys.path.append(base_dir + "lib/python/")
sys.path.append(base_dir + "lib/cpp/")

# other directories
raw_dir = base_dir + "raw/"
processed_dir = base_dir + "processed/"
thesaurus_dir = base_dir + "thesaurus/"

# number of parts the wikipedia file must be splitted to
wiki_preprocessing_split_into = 10

# some checks for consistency
assert os.path.isdir(base_dir)
assert os.path.isdir(raw_dir)
