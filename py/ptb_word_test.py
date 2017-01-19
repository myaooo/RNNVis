"""
The data required for this example is in the data/ dir of the
PTB datasets from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
"""
import tensorflow as tf
import numpy as np

from py.rnn import rnn
from py.rnn.trainer import get_gradient_clipper
from py.datasets.ptb_reader import ptb_raw_data
from py.datasets.data_utils import InputProducer


def test_data_producer(data, batch_size, num_steps):
    # train_data = valid_data
    producer = InputProducer(data, batch_size)
    inputs = producer.get_feeder(num_steps,transpose=True)
    targets = producer.get_feeder(num_steps, offset=1, transpose=True)
    return inputs, targets, targets.epoch_size


def test_lr_decay(global_step):
    base_lr = 1
    if global_step < 7000:
        return base_lr
    else:
        return base_lr * 0.5 ** ((global_step-7000)/2300)

if __name__ == '__main__':

    logdir = './ptb_log'
    train_steps = 20
    batch_size = 20
    epoch_num = 13
    print('Preparing data')
    train_data, valid_data, test_data, vocab_size = ptb_raw_data('./cached_data/simple-examples/data')

    train_inputs, train_targets, epoch_size = test_data_producer(train_data, batch_size, train_steps)
    valid_inputs, valid_targets, _ = test_data_producer(valid_data, batch_size, train_steps)

    # vocab_size = 10000
    model = rnn.RNN('LSTM', tf.random_uniform_initializer(-0.1, 0.1))
    model.set_input([None], tf.int32, vocab_size, embedding_size=200)
    model.add_cell(rnn.BasicLSTMCell, 200)
    model.add_cell(rnn.BasicLSTMCell, 200)
    model.set_output([None, vocab_size], tf.float32)
    model.set_target([None], tf.int32)
    model.set_loss_func(rnn.loss_by_example)
    model.compile()
    print('Start Training')
    model.train(train_inputs, train_targets, train_steps, epoch_size, epoch_num, batch_size, 'GradientDescent', 1.0,
                clipper=get_gradient_clipper('global_norm', 5), keep_prob=1.0, decay=test_lr_decay,
                valid_inputs=valid_inputs, valid_targets=valid_targets, valid_batch_size=batch_size)

    print('Finish Training')
    model.save()
    # test
    # print("start test")
    # j = tf.constant(10, dtype=tf.int32)
    # data = np.array(range(100)).reshape([5,20])
    # data = tf.convert_to_tensor(data)
    # queue = tf.train.range_input_producer(j, shuffle=False)
    # i = queue.dequeue()
    # sect = tf.slice(data, [0, i * 2], [5, 2])
    # sv = tf.train.Supervisor()
    # with sv.managed_session() as sess:
    #     writer = tf.summary.FileWriter("./ptb-model", sess.graph)
    #     # sess.run(tf.global_variables_initializer())
    #     print("start running")
    #     print(sess.run(sect))
    #     print("finish running")

