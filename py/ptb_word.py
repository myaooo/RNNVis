import tensorflow as tf
import numpy as np

from py.rnn import rnn
from py.dataset.ptb_reader import ptb_raw_data
from py.dataset.data_utils import data_feeder, data_batcher

if __name__ == '__main__':

    import cProfile, pstats, io

    pr = cProfile.Profile()
    pr.enable()
    # ... do something ...

    logdir = './ptb_log'
    train_steps = 20
    batch_size = 40
    epoch_num = 1
    print('Preparing data')
    train_data, valid_data, test_data, vocab_size = ptb_raw_data('./dataset/simple-examples/data')
    train_data = valid_data
    batch_len = len(train_data) // batch_size
    epoch_size = (batch_len-1) // train_steps

    train_data = data_batcher(train_data, batch_size)
    valid_data = data_batcher(valid_data, batch_size)
    test_data = data_batcher(test_data, batch_size)

    train_inputs = data_feeder(train_data, train_steps, name='train_inputs')
    train_targets = data_feeder(train_data, train_steps, shift=1, name='train_targets')
    valid_inputs = data_feeder(valid_data, train_steps, name='valid_inputs')
    valid_targets = data_feeder(valid_data, train_steps, shift=1, name='valid_targets')
    # vocab_size = 10000
    model = rnn.RNN('LSTM', tf.random_uniform_initializer(-0.1, 0.1))
    model.set_input([None], tf.int32, vocab_size, embedding_size=256)
    model.add_cell(rnn.BasicLSTMCell, 256)
    model.add_cell(rnn.BasicLSTMCell, 256)
    model.set_output([None, vocab_size], tf.float32)
    model.set_target([None], tf.int32)
    model.set_loss_func(rnn.loss_by_example)
    model.compile()
    print('Start Training')
    model.train(train_inputs, train_targets, train_steps, epoch_size, epoch_num, batch_size, 'Adam', 0.1,
                logdir=logdir)

    print('Finish Training')

    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    with open('profile.txt', 'w+') as f:
        f.write(s.getvalue())
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

