import tensorflow as tf

from py.rnn import rnn

if __name__ == '__main__':

    vocab_size = 1000
    model = rnn.RNN('LSTM')
    model.set_input([None, vocab_size], tf.int32)
    model._add_cell(rnn.BasicLSTMCell, 256)