import cfg
import os

if __name__ ==  "__main__":
    # create the support directories if needed
    if not os.path.exists(cfg.processed_dir):
        os.mkdir(cfg.processed_dir)
    if not os.path.exists(cfg.thesaurus_dir):
        os.mkdir(cfg.thesaurus_dir)

    # create init file in wikiextractor submodule
    with open(cfg.base_dir + "lib/python/wikiextractor/__init__.py") as f:
        pass
