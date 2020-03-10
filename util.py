# coding: utf8
# __author__: tjf141457 (2020/3/9)

import os


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
