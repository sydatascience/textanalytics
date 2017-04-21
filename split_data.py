#! /usr/bin/python3

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
# we are expecting strings in our categorical variables later
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

  new_column = new_column.str.replace(r'(\w)[\'`](\w)', r'\1\2')
  new_column = new_column.str.replace(r'[,;:/\|–?()\[\]{}\'"<>�]', r' ')

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

  # Spelling mistakes
  new_column = new_column.str.replace(r'Excellent?', r'Excellent')
  new_column = new_column.str.replace(r'Operationa(al)?', r'Operational')
  new_column = new_column.str.replace(r'Switzerla(nd)?', r'Switzerland')
  new_column = new_column.str.replace(r'progammer', r'programmer')
  new_column = new_column.str.replace(r'Progamming', r'Programming')
  new_column = new_column.str.replace(r'Registerd', r'Registered')
  new_column = new_column.str.replace(r'Technician?', r'Technician')
  new_column = new_column.str.replace(r'Develpoer', r'Developer')
  new_column = new_column.str.replace(r'Administraotr|Administartor',
                                      r'Administrator')
  new_column = new_column.str.replace(r'Restauarnt', r'Restauarnt')
  new_column = new_column.str.replace(r'Eneigneer', r'Engineer')


  new_column = new_column.str.replace(r'(\d+) %', r'\1%')
  new_column = new_column.str.replace(r'(\w+)(\d+%)', r'\1 \2')
  # Replace some 10%bonus with 10% bonus
  new_column = new_column.str.replace(r'(\d+%)(\w+)', r'\1 \2')

  new_column = new_column.str.replace(r'FinanceCheshire', r'Finance Cheshire')
  new_column = new_column.str.replace(r'ManufacturingCoventry',
                                      r'Manufacturing Coventry')

  new_column = new_column.str.replace(r'RESTAURANTNEWCASTLE',
                                      r'Restaurant Newcastle')
  new_column = new_column.str.replace(r'RESTAURANTHALIFAX',
                                      r'Restaurant Halifax')
  new_column = new_column.str.replace(r'Keynesexciting',
                                      r'Keynes exciting')

  new_column = new_column.str.replace(r'C\+{2}Software',
                                      r'C++ Software')
  new_column = new_column.str.replace(r'PHPunit',
                                      r'PHP unit')



  new_column = new_column.str.replace(r'(?i)(Manager)(\w+)', r'\1 \2')

  # Multi words concated can be split by their capitals.
  new_column = new_column.str.replace(r'([a-z]+)([A-Z][a-z]+)', r'\1 \2')
  new_column = new_column.str.replace(r'Availab(le)?', r'Available')

  new_column = new_column.str.replace(r'Adminstration', r'Administration')
  new_column = new_column.str.replace(r'Adminstrative', r'Administrative')
  new_column = new_column.str.replace(r'recuitment', r'recruitment')
  new_column = new_column.str.replace(r'Proffesional', r'Professional')


  new_column = new_column.str.replace(r'TransformLoad', r'Transform Load')
  new_column = new_column.str.replace(r'Productdeveloper', r'Product Developer')
  new_column = new_column.str.replace(r'BelfastGlasgow', r'Belfast Glasgow')
  new_column = new_column.str.replace(r'PROVIDEDCORK', r'Provided Cork')
  new_column = new_column.str.replace(r'ChildcareNO', r'Childcare NO')
  new_column = new_column.str.replace(r'AfterChildren', r'After Children')
  new_column = new_column.str.replace(r'Homeworking', r'Home Working')
  new_column = new_column.str.replace(r'BallymenaFantastic',
                                      r'Ballymena Fantastic')
  new_column = new_column.str.replace(r'PermLondon', r'Perm London')
  new_column = new_column.str.replace(r'(\w+)(RGN)', r'\1 \2')
  new_column = new_column.str.replace(r'(RGN)(\w+)', r'\1 \2')
  new_column = new_column.str.replace(r'kOTE', r'k OTE')
  new_column = new_column.str.replace(r'Germanspeaking', r'German speaking')
  new_column = new_column.str.replace(r'Availableility', r'Availability')

  return new_column

data['Title'] = clean_free_text_field(data['TitleRaw'])

# Clean FullDescription field.
# Split slashed words. e.g.  "computer/software" with computer software
data["FullDescription"] = data["FullDescription"].str.replace(
    r'(\w+)\/(\w+)', r'\1 \2')

# Remove common recurring pattern
# JobSeeking/SupportEngineerAnalystExchangeWindowsServerAD_job****
data["FullDescription"] = data["FullDescription"].str.replace(
    r'JobSeeking/([\w*]+)_job\*{4}', r'\1')


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

