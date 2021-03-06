"""Library of functions for use with tensorflow notebooks."""
import numpy
import math
import scipy
import sklearn
import nltk


class MeanEstimator(sklearn.base.BaseEstimator):
  """Scikit type estimator that always predicts the training set mean."""
  def fit(self, X, y):
    assert X.shape[0] == y.shape[0]
    self.mean = y.mean()

  def predict(self, X):
    return numpy.full(shape=X.shape[0], fill_value=self.mean)


class PorterTokenizer(object):
  """Tokenizes document and applies porter stemmer to all the words within"""
  def __init__(self):
    self.porter_stemmer = nltk.PorterStemmer()
  def __call__(self, doc):
    word_list = []
    for word in nltk.word_tokenize(doc):
      try:
        word_list.append(self.porter_stemmer.stem(word))
      except IndexError:
        word_list.append(word)
    return word_list


def normalize_input(input_vector, train_mean, train_std):
  """Normalize input vector by minusing training mean and dividing by s.d.

  Args:
    input_vector: numpy.array
    train_mean: float, mean.
    train_std: float, standard deviation.
  Returns:
    numpy.array with approximately mean 0 and s.d. 1
  """
  return (input_vector - train_mean)/train_std


def unnormalize_input(normalized_input_vector, train_mean, train_std):
  """Restor normalized input vector to original scale.

  Args:
    normalized_input: numpy.array with approximately mean 0 and s.d. 1
    train_mean: float, mean.
    train_std: float, standard deviation.
  Returns:
    numpy.array on original scale.
  """
  return (normalized_input_vector * train_std) + train_mean


def unison_shuffled_copies(x, y):
  """Returns a copy of x and y in which the order of the rows has been shuffled.

  Args:
    x: numpy.array
    y: numpy.array
  Returns:
    tuple of two numpy.array objects. The first corresponding to x and the
    second to y
  Raises:
    AssertionError: if x and y are not of the same length.
  """
  assert x.shape[0] == y.shape[0]
  return sklearn.utils.shuffle(x, y)


# Function to generate a training batch.
def generate_batch(batch_size, input_data, input_labels):
  """Creates a generator that generates batches of specified size from the data.

  Args:
    batch_size: int, what size batch to use.
    input_data: X data values.
    input_labels: y data labels.
  Returns:
    Generator that returns tuples of the batch data and the labels.
  """
  batch = numpy.ndarray(
    shape=(batch_size, input_data.shape[1]), dtype=numpy.float32)
  labels = numpy.ndarray(shape=(batch_size, 1), dtype=numpy.float32)
  batch_index = 0
  number_of_batches = math.floor(input_data.shape[0]/batch_size)
  while batch_index < number_of_batches:
    for element_index in range(batch_size):
      # Prevents overflow.
      data_index = ((batch_index * batch_size) + element_index) % input_data.shape[0]
      if type(input_data) == scipy.sparse.csr_matrix:
        batch[element_index] = input_data.getrow(data_index).toarray()
      else:
        batch[element_index] = numpy.array(input_data[data_index])
      labels[element_index] = input_labels[data_index]
    batch_index += 1
    yield batch, labels
