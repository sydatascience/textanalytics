#! /usr/bin/python

import pandas
import numpy
import os

DATA_DIRECTORY = '~/textanalytics/data'
INPUT_DATA_FILE = 'Train_rev1.csv'
TESTING_PROPORTION = .2


# Update working directory
os.chdir(os.path.expanduser(DATA_DIRECTORY))

data = pandas.read_csv(
    os.path.abspath('/'.join([
        os.getcwd(), INPUT_DATA_FILE])))

# Add new transformed/calculated fields
data['DescriptionLength'] = data['FullDescription'].str.len()
data['LogSalaryNormalized'] = numpy.log(data['SalaryNormalized'])

numpy.random.seed(2016)

# Shuffle data set before splitting
data.reindex(numpy.random.permutation(data.index))

rows = numpy.random.binomial(1, 1 - TESTING_PROPORTION, size=len(data)).astype('bool')

# Split data set
training_set = data[rows]
testing_set = data[~rows]

# For writing scripts with limited memory usage for faster iterations
training_set.head(50).to_csv('train_mini.csv')
training_set.head(1000).to_csv('train_medium.csv')

# Write out training and test sets
training_set.to_csv('train.csv')
testing_set.to_csv('test.csv')

