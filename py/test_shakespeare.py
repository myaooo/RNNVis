"""
Tests text_processor and training on tiny shakespeare data
"""


from .procedures import build_model, init_tf_environ
from .datasets.data_utils import InputProducer, split
from .datasets.text_processor import PlainTextProcessor
from tensorflow import flags

flags.DEFINE_string("config_path", None, "The path of the model configuration file")
flags.DEFINE_string("data_path", None, "The path of the input data")
flags.DEFINE_string("log_path", None, "The path to save the log")
FLAGS = flags.FLAGS


def config_path():
    return FLAGS.config_path


def data_path():
    return FLAGS.data_path


def log_path():
    return FLAGS.log_path


def test_data_producer(data, batch_size, num_steps):
    # train_data = valid_data
    producer = InputProducer(data, batch_size)
    inputs = producer.get_feeder(num_steps, transpose=True)
    targets = producer.get_feeder(num_steps, offset=1, transpose=True)
    return inputs, targets, targets.epoch_size

if __name__ == '__main__':

    init_tf_environ()
    print('Building model..')
    model, train_config = build_model(config_path())
    epoch_num = train_config.epoch_num
    keep_prob = train_config.keep_prob
    print('Preparing data..')
    processor = PlainTextProcessor(data_path())
    processor.tag_rare_word(1, model.vocab_size)
    processor.save()
    split_data = split(processor.flat_ids, [0.9, 0.05, 0.05])
    train, valid, test = tuple(split_data)
    # print(valid)

    train_inputs, train_targets, epoch_size = test_data_producer(train, train_config.batch_size, train_config.num_steps)
    valid_inputs, valid_targets, valid_epoch_size = \
        test_data_producer(valid, train_config.batch_size, train_config.num_steps)
    test_inputs, test_targets, test_epoch_size = \
        test_data_producer(test, 1, train_config.num_steps)

    print('Start Training')
    model.train(train_inputs, train_targets, epoch_size, epoch_num,
                valid_inputs=valid_inputs, valid_targets=valid_targets, valid_epoch_size=valid_epoch_size)

    print('Finish Training')
    model.save()
