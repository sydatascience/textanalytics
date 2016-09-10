#! /usr/bin/python
# -*- coding: utf-8 -*-

import pandas
import numpy
import os
import re

DATA_DIRECTORY = '~/textanalytics/data'
INPUT_DATA_FILE = 'Train_rev1.csv'
TESTING_PROPORTION = .3

# TODO (any): Make this check if the required files and data directory exist.
# If not then create/download them.


# Update working directory
os.chdir(os.path.expanduser(DATA_DIRECTORY))

data = pandas.read_csv(
    os.path.abspath('/'.join([
        os.getcwd(), INPUT_DATA_FILE])))

PLACEHOLDER_NA_VALUE = "unknown_value"

# Fill missing values.
# This needs to be done as otherwise null/Nan/NA values will raise errors where
# we are expecting strings in our categorical variables
data["TitleRaw"] = data["Title"].fillna(PLACEHOLDER_NA_VALUE)
data["Company"] = data["Company"].fillna(PLACEHOLDER_NA_VALUE)
data["SourceName"] = data["SourceName"].fillna(PLACEHOLDER_NA_VALUE)
data["ContractTime"] = data["ContractTime"].fillna(PLACEHOLDER_NA_VALUE)
data["ContractType"] = data["ContractType"].fillna(PLACEHOLDER_NA_VALUE)

# Add new transformed/calculated fields
data['DescriptionLength'] = data['FullDescription'].str.len()
data['LogSalaryNormalized'] = numpy.log(data['SalaryNormalized'])

def clean_free_text_field(column):
  # Split slashed words into two distinct tokens
  new_column = column.str.replace(r'(\w+)\/(\w+)', r'\1 \2')
  # Split ampersanded words into two distinct tokens
  new_column = new_column.str.replace(r'(\w{2,})&(\w{2,})', r'\1 \2')

  # Remove all brackets emptying their contents.
  new_column = new_column.str.replace(r'\(([^)]+)\)', r'\1')
  new_column = new_column.str.replace(r'\[([^\]]+)\]', r'\1')

  # Replace Punctuation characters with white space , ; / -
  new_column = new_column.str.replace(r'[,;:/\|–?()\[\]{}\'"<>]', r' ')

  # Split words from "**** pattern"
  new_column = new_column.str.replace(r'([\w]+)(\*{4})', r'\1 \2 ')
  new_column = new_column.str.replace(r'(\*{4})([\w]+)', r'\1 \2 ')

  # Replace all k patterns with capital k. also replace doubles.
  new_column = new_column.str.replace(r'(?i)\*{4}k( \*{4} k)?', r'****K')
  
  new_column = new_column.str.replace(r'(\*{4}K)([\w]+)', r'\1 \2')

  # Correct some words
  new_column = new_column.str.replace(r'bonusbenefits', r'bonus benefits')
  # Remove duplicate spaces
  new_column = new_column.str.replace(r'\s+', r' ')
  return new_column

data['Title'] = clean_free_text_field(data['TitleRaw']) 

# Clean FullDescription field.
# Split slashed words. e.g.  "computer/software" with computer software
data["FullDescription"] = data["FullDescription"].str.replace(r'(\w+)\/(\w+)', r'\1 \2')

#Remove common pattern JobSeeking/SupportEngineerAnalystExchangeWindowsServerAD_job****
data["FullDescription"] = data["FullDescription"].str.replace(r'JobSeeking/([\w*]+)_job\*{4}', r'\1')


# Appending title to full descripton makes sure it is present in all cases. 
data["FullDescriptionWithTitle"] = data["Title"] + " " + data["FullDescription"]


# Remove uncleaned messy salary field.
data.drop('SalaryRaw', axis=1)

numpy.random.seed(2016)

# Shuffle data set before splitting
data.reindex(numpy.random.permutation(data.index))

rows = numpy.random.binomial(1, 1-TESTING_PROPORTION, size=len(data)).astype(
  'bool')

# Split data set
training_set = data[rows]
testing_set = data[~rows]

# For writing scripts with limited memory usage for faster iterations
training_set.head(50).to_csv('train_mini.csv', index=False)
training_set.head(1000).to_csv('train_medium.csv', index=False)
training_set.head(20000).to_csv('train_large.csv', index=False)

# Write out full training and test sets
training_set.to_csv('train.csv', index=False)
testing_set.to_csv('test.csv', index=False)

