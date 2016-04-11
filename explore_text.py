#!/usr/bin/python
"""Docstring
"""
# To default to float division and print function.
from __future__ import (division, print_function)

# Core python libraries
import csv
import re

# External libraries.
import nltk
import numpy
import pandas
import scipy
import sklearn
import sklearn.ensemble

def get_full_text_description(row_list):
  total_description = ' '.join([r['FullDescription'] for r in row_list])
  tokens = nltk.word_tokenize(total_description)
  text = nltk.Text(tokens)
  return text

def get_word_count_dict(text_description):
  word_count = {}
  tokens = nltk.word_tokenize(text_description)
  print(tokens)
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
  return words_vectorized

def main():
  data = pandas.read_csv('data/train_medium.csv').drop(['SalaryRaw'], axis=1)
  X_train_index, X_test_index, Y_train, y_test = sklearn.cross_validation.train_test_split(
      data.index, data['SalaryNormalized'], test_size=.33, random_state=42)

  #This is to keep them as pandas dataframes.
  X_train = data.iloc[X_train_index]
  X_test = data.iloc[X_test_index]

  full_text = X_train['FullDescription']  
  count_vect = sklearn.feature_extraction.text.CountVectorizer(
    stop_words='english', min_df=1)
  X_train_counts = count_vect.fit_transform(full_text)
  Y_train_counts =count_vect.transform(X_test['FullDescription'])
  print(X_train_counts.shape)
  print(Y_train_counts.shape)

  # Guess the average. Create an empty vector of the desired shape.
  average_guess = numpy.empty(y_test.shape)
  average_guess.fill(numpy.mean(Y_train))
  print('Guess value is %s' % average_guess)

  # We want a stochastic gradient descent with l1 norm.
  sgd = sklearn.linear_model.SGDRegressor(penalty='l1', n_iter=100)
  sgd.fit(X_train_counts, Y_train)
  sgd_predictions = sgd.predict(Y_train_counts)

  # Random Forest.
  rf = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
  rf.fit(X_train_counts, Y_train)
  rf_predictions = rf.predict(Y_train_counts)

  # Mean Absolute Error
  average_guess_mae = sklearn.metrics.mean_absolute_error(y_test, average_guess)
  sgd_mae = sklearn.metrics.mean_absolute_error(y_test, sgd_predictions)
  rf_mae = sklearn.metrics.mean_absolute_error(y_test, rf_predictions)
  print('Guess the average Mean Absolute Error: {:10.4f}'.format(average_guess_mae))
  print('SGDRegressor Mean Absolute Error: {:10.4f}'.format(sgd_mae))
  print('Random Forest Regressor Mean Absolute Error: {:10.4f}'.format(rf_mae))


if __name__ == "__main__":
    # execute only if run as a script
    main()
