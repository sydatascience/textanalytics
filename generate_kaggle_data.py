#! /usr/bin/python3
# Core python libraries
import csv
import re
import math
import os
import time

# External libraries.
import nltk
import numpy
import pandas
import scipy
import sklearn
import tempfile
import pickle

import tf_utils


def process_text_column(train_dataframe, test_dataframe, colname, vectorizer):
    print("Starting to process \"%s\" text column." % colname)
    train_output = vectorizer.fit_transform(train_dataframe[colname])
    test_output = vectorizer.transform(test_dataframe[colname])
    print("\"%s\" number of columns = %d" % (colname, train_output.shape[1]))
    return (train_output, test_output)


def process_categorical_column(
        train_dataframe, test_dataframe, colname, label_encoder, one_hot_encoder):
    print("Starting to process \"%s\" categorical column." % colname)
    column = train_dataframe[colname].values
    test_column = test_dataframe[colname].values

    labels = label_encoder.fit_transform(column).reshape(-1, 1)
    train_output_values = one_hot_encoder.fit_transform(labels)


    test_output_labels = label_encoder.transform(test_column).reshape(-1, 1)
    test_output_values = one_hot_encoder.transform(test_output_labels)

    print("\"%s\" number of columns = %d" % (colname, train_output_values.shape[1]))
    return (train_output_values, test_output_values)


def process_categorical_columns(
        train_dataframe, test_dataframe, columns, label_encoder, one_hot_encoder):
    print("Starting to process all the categorical columns.")
    train_data = []
    test_data = []

    for colname in columns:
      train_columns, test_columns = process_categorical_column(
        train_dataframe, test_dataframe, colname, label_encoder, one_hot_encoder)
      train_data.append(train_columns)
      test_data.append(test_columns)
    return scipy.sparse.hstack(train_data), scipy.sparse.hstack(test_data)


def generate_kaggle_data(min_word_frequency, X_train, X_test):

    tfidf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
        min_df=min_word_frequency, decode_error='ignore',
        tokenizer=tf_utils.PorterTokenizer())

    count_vectorizer = sklearn.feature_extraction.text.CountVectorizer(
        min_df=min_word_frequency, decode_error='ignore')

    #Count vectorizer this one, no stemming.
    location_raw_train, location_raw_test = process_text_column(
        X_train, X_test, "LocationRaw", count_vectorizer)

    # Text columns:
    title_train, title_test = process_text_column(X_train, X_test, "Title", tfidf_vectorizer)

    # This one takes a while: Pickle this bugger.
    print("Processing the \"FullDescription\" field will take a while")
    full_description_train, full_description_test = process_text_column(
        X_train, X_test, "FullDescription", tfidf_vectorizer)


    # Categorical columns: (category, contract, and source, were represented using a 1-of-K encoding.)
    # TODO(Max): Fix this the SKLEARN way
    
    label_encoder = sklearn.preprocessing.LabelEncoder()
    one_hot_encoder = sklearn.preprocessing.OneHotEncoder()

    #category_train, category_test = process_categorical_column_dict(X_train, X_test, "Category", dict_vectorizer)
    # Hmmm... Company appears to add over 16k columns
    cols_to_transform = ['Category', 'ContractType', 'ContractTime', 'SourceName']
    categorical_columns_sparse_matrix_train, categorical_columns_sparse_matrix_test = process_categorical_columns(
        X_train, X_test, cols_to_transform, label_encoder, one_hot_encoder)


    #print(sparse_df_train.columns.values)
    print(categorical_columns_sparse_matrix_train.shape)
    print(categorical_columns_sparse_matrix_test.shape)
    # Why not company?

    # Stack all the sparse datasets.

    # Unsure if I need this one.
    normalizer = sklearn.preprocessing.Normalizer()

    x_train = normalizer.fit_transform(scipy.sparse.csr_matrix(scipy.sparse.hstack(
        [title_train, location_raw_train, full_description_train, categorical_columns_sparse_matrix_train])
                                                                                                     ))
    x_test = normalizer.transform(scipy.sparse.csr_matrix(scipy.sparse.hstack(
        [title_test, location_raw_test, full_description_test, categorical_columns_sparse_matrix_test])
                                                              ))
    print("x_train shape: %s, %s" % x_train.shape)
    print("x_test shape: %s, %s" % x_test.shape)

    # Need this for my batch code below
    assert type(x_train) == scipy.sparse.csr_matrix
    assert type(x_test)== scipy.sparse.csr_matrix
    return x_train, x_test

if __name__ == "__main__":
  min_word_frequency = 25
  data = pandas.read_csv("data/train.csv")
  TEST_SIZE = .2
  LABEL_COLUMN = "SalaryNormalized"

  # sklearn.cross_validation has been replaced with model_selection.
  X_train_index, X_test_index, Y_train, Y_test = (
      sklearn.model_selection.train_test_split(
          data.index, data[LABEL_COLUMN], test_size=TEST_SIZE, random_state=42))

  # Keep train and test as pandas dataframes.
  X_train = data.iloc[X_train_index]
  X_test = data.iloc[X_test_index]
  kaggle_file = "kaggle_output.pkl"
  x_train, x_test = generate_kaggle_data(
      min_word_frequency, X_train, X_test)
  print("Data generated. Writing pickle.")
  pickle.dump({'x_train': x_train, 'x_test':x_test}, open(kaggle_file, 'wb'))

