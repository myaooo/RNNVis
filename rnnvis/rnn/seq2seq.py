import tensorflow as tf

from rnnvis.vendor.seq2seq_model import Seq2SeqModel

from rnnvis.rnn.rnn import RNNModel, RNN, DropOutWrapper, MultiRNNCell, _input_and_global
from rnnvis.rnn.varlen_support import sequence_length, last_relevant
from rnnvis.rnn.command_utils import data_type
from rnnvis.rnn.eval_recorder import Recorder


class Seq2SeqEvaluator():

    def __init__(self, model, batch_size=1, num_steps=1, record_every=1, log_state=True, log_input=False,
                 log_output=True, log_gradients=False, log_pos=False, dynamic=True):
        """
        Create an unrolled rnn model with TF tensors
        :param rnn:
        :param batch_size:
        :param num_steps:
        :param keep_prob:
        :param name:
        """
        assert isinstance(model, Seq2SeqModel)
        self.model = model
        self.batch_size = batch_size
        self.num_steps = num_steps

        self.dynamic = dynamic
        self.current_state = None

    def evaluate_and_record(self, sess, inputs, targets, recorder, verbose=True, refresh_state=False):
        """
        A similar method like evaluate.
        Evaluate model's performance on a sequence of inputs and targets,
        and record the detailed information with recorder.
        :param inputs: an object convertible to a numpy ndarray, with 2D shape [batch_size, length],
            elements are word_ids of int type
        :param targets: same as inputs, no loss will be calculated if targets is None
        :param sess: the sess to run the computation
        :param recorder: an object with method `start(inputs, targets)` and `record(record_message)`
        :param verbose: verbosity
        :return:
        """

        assert isinstance(inputs, Feeder), 'expect inputs type Feeder but got type {:s}'.format(str(type(inputs)))
        assert isinstance(targets, Feeder) or targets is None
        assert isinstance(recorder, Recorder), "recorder should be an instance of rnn.eval_recorder.Recorder!"
        recorder.start(inputs, targets, self.pos_tagger)
        input_size = inputs.epoch_size
        print("input size: {:d}".format(input_size))
        eval_ops = self.summary_ops
        self.model.reset_state()
        for i in range(0, input_size, self.record_every):
            if refresh_state:
                self.model.reset_state()
            n_steps = input_size - i if i + self.record_every > input_size else self.record_every

            evals, _ = self.model.run(inputs, targets, n_steps, sess, eval_ops=eval_ops,
                                      verbose=False, refresh_state=False)
            messages = [{name: value[i] for name, value in evals.items()} for i in range(n_steps)]
            for message in messages:
                recorder.record(message)
            if verbose and (i//self.record_every + 1) % (input_size // self.record_every // 10) == 0:
                print("[{:d}/{:d}] completed".format(i+self.record_every, input_size), flush=True)
        recorder.flush()
        print("Evaluation done!")