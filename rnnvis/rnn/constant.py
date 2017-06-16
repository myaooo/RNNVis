import tensorflow as tf

BasicRNNCell = tf.contrib.rnn.BasicRNNCell
BasicLSTMCell = tf.contrib.rnn.BasicLSTMCell
LSTMCell = tf.contrib.rnn.LSTMCell
GRUCell = tf.contrib.rnn.GRUCell
MultiRNNCell = tf.contrib.rnn.MultiRNNCell
DropOutWrapper = tf.contrib.rnn.DropoutWrapper
# EmbeddingWrapper = tf.contrib.rnn.EmbeddingWrapper
InputProjectionWrapper = tf.contrib.rnn.InputProjectionWrapper
OutputProjectionWrapper = tf.contrib.rnn.OutputProjectionWrapper
LSTMStateTuple = tf.contrib.rnn.LSTMStateTuple

tf.GraphKeys.INPUTS = 'my_inputs'