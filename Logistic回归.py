#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:zbt
# datetime:2019/3/27 21:38
# software: PyCharm
from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
import numpy as np
a=['alt.atheism']
newsgroups_train = fetch_20newsgroups(subset = 'train',categories = a) #不能直接写categories
pprint(list(newsgroups_train.target_names))
pprint(newsgroups_train.filenames.shape)
pprint(newsgroups_train.target.shape)
