import collections

import tensorflow as tf
import tensorflow.contrib.rnn as rnn

RNNCell = rnn.RNNCell
BasicRNNCell = rnn.BasicRNNCell
BasicLSTMCell = rnn.BasicLSTMCell
LSTMCell = rnn.LSTMCell
GRUCell = rnn.GRUCell
MultiRNNCell = rnn.MultiRNNCell
DropOutWrapper = rnn.DropoutWrapper
EmbeddingWrapper = rnn.EmbeddingWrapper
InputProjectionWrapper = rnn.InputProjectionWrapper
OutputProjectionWrapper = rnn.OutputProjectionWrapper
LSTMStateTuple = rnn.LSTMStateTuple
static_rnn = rnn.static_rnn
# rnn.EmbeddingWrapper

tf.GraphKeys.INPUTS = 'my_inputs'
