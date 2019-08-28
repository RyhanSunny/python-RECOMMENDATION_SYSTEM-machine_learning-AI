import numpy as np
from lightfm.datasets import fetch_movielens # movielens is a data set of 100k real movie ratings
from lightfm import LightFM # lightfm performs recommendation algorithm

# fetching data and formatting it
data = fetch_movielens(min_rating = 4.0) # CSV file

# the fetch_movielens method will create an INTERACTION MATRIX from
# our CSV file and store it in our data variable as a DICTIONARY

# print-training and testing
print(repr(data['print']))
print(repr(data('test')))

# our fetch_movielens method splits our data set into 'training' and 'testing' data
# and we can retrieve each by using the 'train' and 'test'strings as KEYS (dictionary keys)
# we printed out both above

# todo: install lightfm again, run the code, store our model in a variable
#




