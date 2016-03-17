#!/usr/bin/python
"""Docstring
"""
# To default to float division
from __future__ import division

import csv
import re
import nltk
import sklearn
import numpy

def get_formatted_csv(filename):
  with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    return_list = []
    for row_index, row in enumerate(csvreader):
      if row_index == 0:
        header = row
      else:
        value_dict = {}
        for column_index, value in enumerate(row):
          value_dict[header[column_index]] = value
        return_list.append(value_dict)
    return return_list

def get_full_text_description(row_list):
  total_description = ' '.join([r['FullDescription'] for r in row_list])
  tokens = nltk.word_tokenize(total_description)
  text = nltk.Text(tokens)
  return text

def get_word_count_dict(text_description):
  word_count = {}
  tokens = nltk.word_tokenize(text_description)
  print tokens
  for word in tokens:
    word_count[word] = word_count.get(word, 0) + 1
  return word_count

def get_training_set(input_data):
  y = numpy.array([row['SalaryNormalized'] for row in input_data])
  x = [get_word_count_dict(row['FullDescription']) for row in input_data]
  return x, y

def get_vocab(text):
  words = [word.lower() for word in text]
  vocab = sorted(set(words))
  return vocab

def convert_dict_list_to_sparse_matrix(dict_list):
  vec = sklearn.feature_extraction.DictVectorizer()
  words_vectorized = vec.fit_transform(dict_list)
  print vec.get_feature_names()
  return words_vectorized

def fit_sgd_regressor(x, y):
  clf = sklearn.linear_model.SGDRegressor()
  clf.fit(x, y)
  return clf

def main():
  formatted_data = get_formatted_csv('data/train_mini.csv')
  full_text = get_full_text_description(formatted_data)
  x, y = get_training_set(formatted_data)
  new_x = convert_dict_list_to_sparse_matrix(x)

  sgd = fit_sgd_regressor(new_x, y)
  print sgd
  print y
  print sgd.predict(new_x)

if __name__ == "__main__":
    # execute only if run as a script
    main()
