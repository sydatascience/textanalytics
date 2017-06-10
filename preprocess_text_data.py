# External libraries.
import numpy
import pandas
import scipy
import sklearn
# I am not sure why this was throwing an error. But importing it specifically worked.
from sklearn.feature_extraction.text import CountVectorizer

def get_data(file_name):
  return pandas.read_csv('data/%s' % file_name)


def get_lda_training_data(file_name, column_name, min_word_frequency=1):
  """Fetches the required bag of words representation of the desired file.

  Args:
    file_name: (str) filename found in the data folder.
    column_name: (str) name of the column to create bag of words on.
    min_word_frequency: (int) minimum number of times a word must be seen.
  Returns:
    (tuple) of (numpy.array) Bag of words representation sparse matrix and a
        list of the words represented.
  """
  data = get_data(file_name)
  column = data[column_name]

  count_vect = CountVectorizer(
    stop_words='english', min_df=min_word_frequency, analyzer='word')

  bag_of_words = count_vect.fit_transform(column)


  return (bag_of_words, count_vect.get_feature_names())

def get_bow_untransformed(file_name, column_name, min_word_frequency=1):
  """Fetches the required bag of words representation of the desired file.

  Args:
    file_name: (str) filename found in the data folder.
    column_name: (str) name of the column to create bag of words on.
    min_word_frequency: (int) minimum number of times a word must be seen.
  Returns:
    (tuple) of (numpy.array) Bag of words representation sparse matrix and a
        list of the words represented.
  """
  data = get_data(file_name)
  column = data[column_name]

  count_vect = CountVectorizer(
    stop_words='english', min_df=min_word_frequency, analyzer='word')
 
  return (bag_of_words, count_vect.get_feature_names())

def create_job_listing_string(file_name):
  template = """\nJob Title: %s\nLocation: %s\nJob Description: %s\n"""

  data = get_data(file_name)
  complete_listing = ""
  for row_index in range(len(data)):
    row = data.iloc[row_index]
    complete_listing += template % (row["Title"],
                                    row['LocationNormalized'],
                                    row["FullDescription"])
  with open("data/complete_listing.txt", "w+") as text_file:
    text_file.write(complete_listing)
  return complete_listing

#create_job_listing_string("train.csv")
