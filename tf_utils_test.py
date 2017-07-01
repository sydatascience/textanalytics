#!/usr/bin/python3
import unittest
import numpy
import scipy
import tf_utils

class TestTfUtils(unittest.TestCase):
  def test_normalize_input(self):
    input_vector = numpy.array([2, 2, 3, 2, 2])
    input_mean = input_vector.mean()
    input_stddev = input_vector.std()
    normalized_input = tf_utils.normalize_input(input_vector, input_mean, input_stddev)
    expected_result = numpy.array([-.5, -.5, 2, -.5, -.5])
    numpy.testing.assert_array_almost_equal(normalized_input, expected_result)

  def test_unnormalize_input(self):
    input_vector = numpy.array([2, 2, 3, 2, 2])
    normalized_vector = numpy.array([-.5, -.5, 2, -.5, -.5])
    input_mean = input_vector.mean()
    input_stddev = input_vector.std()
    unnormalized_output = tf_utils.unnormalize_input(
        normalized_vector, input_mean, input_stddev)
    numpy.testing.assert_array_almost_equal(unnormalized_output, input_vector)

  def test_normalize_unnormalize_input(self):
    input_vector = numpy.arange(7)
    input_mean = input_vector.mean()
    input_stddev = input_vector.std()
    normalized_input = tf_utils.normalize_input(input_vector, input_mean, input_stddev)
    unnormalized_output = tf_utils.unnormalize_input(
        normalized_input, input_mean, input_stddev)
    numpy.testing.assert_array_equal(unnormalized_output, input_vector)

  def test_unison_shuffled_copies(self):
    input_x = numpy.array([[1,2],[3,4],[5,6]])
    input_y = numpy.array([1,3,5])
    numpy.random.seed(421)
    output_x, output_y = tf_utils.unison_shuffled_copies(input_x, input_y)
    numpy.testing.assert_array_equal(output_x, numpy.array([[1,2],[5,6],[3,4]]))
    numpy.testing.assert_array_equal(output_y, numpy.array([1,5,3]))

  def test_unison_shuffled_copies(self):
    input_x = numpy.array([[3, 1, 6],[9, 4, 1],[5, 6, 2],[7, 9, 6]])
    input_y = numpy.array([1, 3, 5, 11])
    numpy.random.seed(42)
    output_x, output_y = tf_utils.unison_shuffled_copies(input_x, input_y)
    expected_x = numpy.array([[9, 4, 1],[7, 9, 6],[3, 1, 6],[5, 6, 2]])
    expected_y = numpy.array([3, 11, 1, 5])
    numpy.testing.assert_array_equal(output_x, expected_x)
    numpy.testing.assert_array_equal(output_y, expected_y)

  def test_generate_batch_dense_matrix(self):
    input_x = numpy.array([[3, 1, 6],[9, 4, 1],[5, 6, 2],[7, 9, 6]])
    input_y = numpy.array([1, 3, 5, 11])
    batch_generator = tf_utils.generate_batch(1, 4, 3, input_x, input_y)
    for i in range(len(input_x)):
      x, y = next(batch_generator)
      numpy.testing.assert_array_equal(x, input_x[i].reshape(1,3))
      numpy.testing.assert_array_equal(y, input_y[i])

  def test_generate_batch_sparse_matrix(self):
    input_x = scipy.sparse.csr_matrix([[3, 0, 0],[9, 4, 0],[0, 0, 2],[7, 9, 6]])
    input_y = numpy.array([1, 3, 5, 11])
    batch_generator = tf_utils.generate_batch(1, 4, 3, input_x, input_y)
    for i in range(input_x.shape[0]):
      x, y = next(batch_generator)
      numpy.testing.assert_array_equal(x, input_x.getrow(i).toarray())
      numpy.testing.assert_array_equal(y, input_y[i])


if __name__ == '__main__':
  unittest.main()
