#! /usr/bin/python3

# External libraries.
import numpy
import pandas
import scipy
import sklearn
import tensorflow as tf

#Note to self enable gpu support.

#Get input data.
file_name = 'train_mini.csv'
column_name = 'FullDescription'

data = pandas.read_csv('data/%s' % file_name)
column = data[column_name]

max_length = column.str.len().max()
print("Max length column == %s" % max_length)

#This needs to change
train_input = column
train_output = column

#Convert to binary character vectors

#Construct Model

# dimensions for data are [Batch Size, Sequence Length, Input Dimension]
data = tf.placeholder(tf.float32, [None, max_length, 1])

# Unsure about this.
target = tf.placeholder(tf.float32, [None, max_length+1])

# too high a value may lead to overfitting or a very low value may yield
# extremely poor results.
num_hidden = 24
cell = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)
val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([num_hidden,
                                          int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,
                                                                1e-10,1.0)))
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)

mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))


#Run model
init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

#Profit
batch_size = 1000
no_of_batches = int(len(train_input)/batch_size)
epoch = 500
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
    print("Epoch - ", str(i))
#incorrect = sess.run(error,{data: test_input, target: test_output})
#print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
sess.close()

