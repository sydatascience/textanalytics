"""Library of functions for use with tensorflow notebooks."""
import numpy
import sklearn

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
def generate_batch(batch_size, number_of_batches, max_document_length,
                   input_data, input_labels, sparse=True):
  """Creates a generator that generates batches of specified size from the data.

  Args:
    batch_size: int, what size batch to use.
    number_of_batches: int, how many batches are generator should provide.
    max_document_length: int, how long the longest input document was.
    input_data: X data values.
    input_labels: y data labels.
    sparse: bool of whether we are using a sparse matrix.
  Returns:
    Generator that returns tuples of the batch data and the labels.
  """
  batch = numpy.ndarray(
    shape=(batch_size, max_document_length), dtype=numpy.int32)
  labels = numpy.ndarray(shape=(batch_size, 1), dtype=numpy.float32)
  batch_index = 0
  while batch_index < number_of_batches:
    for element_index in range(batch_size):
      # Prevents overflow.
      data_index = ((batch_index * batch_size) + element_index) % input_data.shape[0]
      if sparse:
        batch[element_index] = input_data.getrow(data_index).toarray()
      else:
        batch[element_index] = numpy.array(input_data[data_index])
      labels[element_index] = input_labels[data_index]
    batch_index += 1
    yield batch, labels
