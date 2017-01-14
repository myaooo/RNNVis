import tensorflow as tf
import numpy as np

from py.rnn import rnn

if __name__ == '__main__':

    # vocab_size = 1000
    # model = rnn.RNN('LSTM')
    # model.set_input([None, vocab_size], tf.int32)
    # model._add_cell(rnn.BasicLSTMCell, 256)

    # test
    print("start test")
    j = tf.constant(10, dtype=tf.int32)
    data = np.array(range(100)).reshape([5,20])
    data = tf.convert_to_tensor(data)
    queue = tf.train.range_input_producer(j, shuffle=False)
    i = queue.dequeue()
    sect = tf.slice(data, [0, i * 2], [5, 2])
    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
        writer = tf.summary.FileWriter("./ptb-model", sess.graph)
        # sess.run(tf.global_variables_initializer())
        print("start running")
        print(sess.run(sect))
        print("finish running")