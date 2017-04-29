import nltk
import random
import sys
import os
import glob
import pickle
import re
import pennToWordnet as pwn
import locale
import WordShapeClassifer as ws
import itertools
from collections import Counter
import numpy as np
import sklearn as sk
from sklearn import svm
from gensim.models import Word2Vec
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import ClusterCentroids, CondensedNearestNeighbour
from imblearn.ensemble import EasyEnsemble
from datetime import datetime as dt
import matplotlib.pyplot as plt
from itertools import cycle
import time
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


LOAD_DOCS = True
LOAD_FEATS = True
WORD2VEC = True

WED = 10 # Word Embedding Dimension

## Regex constants ##

INITCAP = 101
ALLCAPS = 102
CAPSMIX =103
HASDIGIT = 104
SINGLEDIGIT = 105
DOUBLEDIGIT = 106
FOURDIGITS = 107
NATURALNUM = 108
REALNUM = 109
ALPHANUM = 110
HASDASH = 111
PUNCTUATION = 112
PHONE1 = 113
PHONE2 = 114
FIVEDIGITS = 115
NOVOWELS = 116
HASDASHNUMALPHA = 117
DATESEPARATOR = 118

