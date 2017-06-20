import tensorflow as tf

from rnnvis.vendor.seq2seq_model import Seq2SeqModel

from rnnvis.rnn.rnn import RNNModel, RNN, DropOutWrapper, MultiRNNCell, _input_and_global
from rnnvis.rnn.varlen_support import sequence_length, last_relevant
from rnnvis.rnn.command_utils import data_type
from rnnvis.rnn.eval_recorder import Recorder

from rnnvis.datasets.data_utils import Feeder


class Seq2SeqEvaluator():

    def __init__(self, model, batch_size=1, record_every=1, log_state=True, log_input=False,
                 log_output=True, log_pos=False):
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
        self.record_every = record_every
        self.log_state = log_state
        self.log_input = log_input
        self.log_output = log_output

        self.current_state = None

        self.pos_tagger = None
        if log_pos:
            # if self.model.id_to_word is None:
            #     raise ValueError('Evaluator: RNN instance needs to have id_to_word property in order to log_pos!')
            import nltk  # lazy import

            def tagger(ids):
                tokens = model.get_word_from_id(ids)
                if len(tokens) != len(ids):
                    raise ValueError('Evaluator: tokens length {:d} and ids length {:d} mismatch'
                                     .format(len(tokens), len(ids)))
                tokens_tags = nltk.pos_tag(tokens, tagset='universal')
                _, tags = zip(*tokens_tags)
                return tags

            self.pos_tagger = tagger

    def evaluate_and_record(self, sess, data, recorder, verbose=True, refresh_state=False):
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

        assert isinstance(recorder, Recorder), "recorder should be an instance of rnn.eval_recorder.Recorder!"
        inputs, targets = zip(*data)
        recorder.start(inputs, targets, self.pos_tagger)
        input_size = len(data)
        print("input size: {:d}".format(input_size))
        # self.model.reset_state()
        for i, batch_data in self.model.get_batches(data):
            _, loss, outputs, states = \
                self.model.step(sess, batch_data[0], batch_data[1], batch_data[2], forward_only=True)
            raw_message = []
            if(self.log_output):

            encoder_states = states[:self.model.encoder_size]
            decoder_states = states[self.model.encoder_size:]
            n_steps = input_size - i if i + self.record_every > input_size else self.record_every

            messages = [{name: value[i] for name, value in evals.items()} for i in range(n_steps)]
            for message in messages:
                recorder.record(message)
            if verbose and (i//self.record_every + 1) % (input_size // self.record_every // 10) == 0:
                print("[{:d}/{:d}] completed".format(i+self.record_every, input_size), flush=True)
        recorder.flush()
        print("Evaluation done!")