import nltk
import random
import sys
import os
import glob
import pickle

## Regex constants ##

INITCAP = "init cap"
ALLCAPS = "all caps"
CAPSMIX = "caps mix"
HASDIGIT = "has digit"
SINGLEDIGIT = "single digit"
DOUBLEDIGIT = "double digit"
FOURDIGITS = "four digit"
NATURALNUM = 'natural number'
REALNUM = 'real number'
ALPHANUM = 'alpha number'
HASDASH = 'has dash'
PUNCTUATION = 'punctuation'
PHONE1 = 'phone 1'
PHONE2 = 'phone 2'
FIVEDIGIT = 'five digit'
NOVOWELS = 'no vowels'
HASDASHNUMALPHA = 'has dash num alpha'
DATESEPARATOR = 'date separator'
