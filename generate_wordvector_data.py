#!/usr/bin/python3
""""""
# Forked and modified from https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
import collections
import math
import os
import random
import zipfile
import re
import time

import pandas
import nltk

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import gensim


print("Done")


target_column = 'CompleteJobListing'

input_data = pandas.read_csv('data/train.csv') #Can read a subset. First nrows of the total.

# Continue from down below.


# Load Google's pre-trained Word2Vec model.
word2vec = gensim.models.KeyedVectors.load_word2vec_format('./data/wordvectors/GoogleNews-vectors-negative300.bin', binary=True)  

# Can fix some spelling mistakes and or british spellings
def get_word_vec(word):
    if type(word2vec) == dict:
        return word2vec.get(word, word2vec['UNK'])
    else:
        try:
            return word2vec.word_vec(word)
        except KeyError:
            try:
                #Try lower case
                return word2vec.word_vec(word.lower())
            except KeyError:
                # Collect these in some class member variable for examination.
                print("Word '%s' not found." % word)


# Include UNK not used.
def convert_listing(job_listing, word_2_vec_dict=word2vec, include_unk=True):
    array_list = []
    for word in nltk.word_tokenize(job_listing):
        vector = get_word_vec(word)
        if vector is not None:
            array_list.append(vector)
    return numpy.average(array_list, axis=0)

def convert_all_listings(job_listing_column, word_2_vec_dict=word2vec, include_unk=True):
    output_list = []
    for job_listing in job_listing_column:
        output_list.append(convert_listing(job_listing))
    return numpy.stack(output_list)


text_data = input_data[target_column]

#clean_cell(example)

word_2_vec_data = convert_all_listings(text_data)

print(word_2_vec_data.shape)
print(word_2_vec_data)


target_value = numpy.array(input_data['SalaryNormalized'])
numpy.savetxt("word2vectargets.csv", target_value, delimiter=",")
numpy.savetxt("word2vecdf.csv", word_2_vec_data, delimiter=",")
