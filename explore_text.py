#!/usr/bin/python
"""Docstring
"""
# To default to float division
from __future__ import division

import csv
import re
import nltk


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

def get_vocab(text):
  words = [word.lower() for word in text]
  vocab = sorted(set(words))
  return vocab


def main():
  formatted_data = get_formatted_csv('train_mini.csv')
  full_text = get_full_text_description(formatted_data)
  print get_vocab(full_text)


if __name__ == "__main__":
    # execute only if run as a script
    main()
