
from tensorflow import flags

from rnnvis.procedures import build_model, init_tf_environ, pour_data
from rnnvis.datasets.data_utils import get_sp_data_producer
from rnnvis.db.sentiment_prediction import get_datasets_by_name


flags.DEFINE_string("config_path", None, "The path of the model configuration file")
flags.DEFINE_string("data_name", None, "The name of the datasets in db")
flags.DEFINE_integer('gpu_num', 0, "The number of the gpu to use, 0 to use no gpu.")
FLAGS = flags.FLAGS


def config_path():
    return FLAGS.config_path


if __name__ == '__main__':

    init_tf_environ(FLAGS.gpu_num)
    print('Building model..')
    model, train_config = build_model(config_path())
    print('Adding Evaluator')
    model.add_evaluator(1, 1, True, True, True)
    epoch_num = train_config.epoch_num
    keep_prob = train_config.keep_prob
    batch_size = train_config.batch_size
    num_steps = train_config.num_steps

    print('Preparing data..')
    producers = pour_data(str(FLAGS.data_name), ['train', 'valid', 'test'], batch_size, num_steps)
    train_inputs, train_targets, epoch_size = producers[0]
    valid_inputs, valid_targets, valid_epoch_size = producers[1]
    test_inputs, test_targets, test_epoch_size = producers[2]

    print('Start Training')
    model.train(train_inputs, train_targets, epoch_size, epoch_num,
                valid_inputs=valid_inputs, valid_targets=valid_targets, valid_epoch_size=valid_epoch_size)

    print('Finish Training')
    model.save()
    print('Testing...')
    model.validate(test_inputs, test_targets, test_epoch_size)

    print('Evaluating sentences')
