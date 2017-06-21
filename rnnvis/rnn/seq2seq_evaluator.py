import math

import numpy as np

from rnnvis.rnn.seq2seq import Seq2SeqModel
from rnnvis.rnn.eval_recorder import Recorder
from rnnvis.datasets.data_utils import Feeder


class Seq2SeqFeeder(Feeder):
    def __init__(self, data, batch_size, epoch_num=None):
        self.i = 0
        self.data = data
        self.max_length = len(data[0])
        self.num_steps = self.max_length
        self.batch_size = batch_size
        self.max_epoch_num = math.inf if epoch_num is None else int(epoch_num)
        self.epoch_num = 0
        self._epoch_size = len(data) // batch_size
        self._shape = [batch_size, self.num_steps]

    def deque(self, transpose=None):
        if transpose is not None:
            raise NotImplementedError("Seq2SeqFeeder has no transpose option!")
        _data = self.top
        self.step(1)
        return _data

    @property
    def top(self):
        start = self.i * self.batch_size
        end = start + self.batch_size
        return self.data[start:end, :]

    @property
    def shape(self):
        return self._shape

    @property
    def epoch_size(self):
        return self._epoch_size

    @property
    def full_data(self):
        return self.data

    @property
    def need_refresh(self):
        return True

    def step(self, k):
        self.i += k
        if self.i >= self.epoch_size:
            self.i -= self.epoch_size
            self.epoch_num += 1
        if self.epoch_num > self.max_epoch_num:
            raise ValueError("Exceeds maximum epoch num!")


class Seq2SeqEvaluator():

    def __init__(self, model, log_state=True,
                 log_output=True, log_pos=False):
        """
        Create an unrolled rnn model with TF tensors
        """
        assert isinstance(model, Seq2SeqModel)
        self.model = model
        # self.batch_size = batch_size
        # self.record_every = record_every
        self.log_state = log_state
        # self.log_input = log_input
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

    @property
    def batch_size(self):
        return self.model.batch_size

    def evaluate_and_record(self, data, recorders, verbose=True):
        """
        A similar method like evaluate.
        Evaluate model's performance on a sequence of inputs and targets,
        and record the detailed information with recorder.
        :param recorders: an object with method `start(inputs, targets)` and `record(record_message)`
        :param verbose: verbosity
        :return:
        """

        assert isinstance(recorders[0], Recorder), "recorder0 should be an instance of rnn.eval_recorder.Recorder!"
        assert isinstance(recorders[1], Recorder), "recorder1 should be an instance of rnn.eval_recorder.Recorder!"
        model = self.model
        def pack_data(data):
            batch_encoder_inputs, batch_decoder_inputs, batch_weights = \
                zip(*model.get_batches(data))
            batch_encoder_inputs = np.hstack(batch_encoder_inputs).T
            batch_decoder_inputs = np.hstack(batch_decoder_inputs).T
            batch_weights = np.hstack(batch_weights).T
            return batch_encoder_inputs, batch_decoder_inputs, batch_weights

        encoder_inputs, decoder_inputs, weights = pack_data(data)
        inputs, targets = zip(*data)

        recorders[0].model_name += '-encoder'
        recorders[1].model_name += '-decoder'
        if self.log_state:
            input_feeder = Seq2SeqFeeder(encoder_inputs, self.batch_size)
            recorders[0].start(input_feeder, None, self.pos_tagger)
        if self.log_state or self.log_output:
            target_feeder = Seq2SeqFeeder(decoder_inputs, self.batch_size)
            recorders[1].start(target_feeder, None, self.pos_tagger)
        input_size = len(data)
        print("input size: {:d}".format(input_size))
        # self.model.reset_state()
        for i, batch_data in enumerate(self.model.get_batches(data)):
            _, loss, outputs, states = \
                self.model.step(batch_data[0], batch_data[1], batch_data[2], forward_only=True)
            messages1 = {}
            messages2 = {}
            if self.log_output:
                # messages1['output'] = outputs
                messages2['output'] = outputs
            if self.log_state:
                encoder_states = states[:self.model.encoder_size]
                decoder_states = states[self.model.encoder_size:]
                messages1['state_h'] = encoder_states
                messages2['state_h'] = decoder_states

            if len(messages1) > 0:
                messages1 = [{key: value[i] for key, value in messages1.items()} for i in range(self.model.encoder_size)]
                for message in messages1:
                    recorders[0].record(message)
            if len(messages2) > 0:
                messages2 = [{key: value[i] for key, value in messages2.items()} for i in range(self.model.decoder_size)]
                for message in messages2:
                    recorders[1].record(message)
            if verbose and (i + 1) % (input_size // self.batch_size // 10) == 0:
                print("[{:d}/{:d}] completed".format((i+1)*self.batch_size, input_size), flush=True)
        recorders[0].flush()
        recorders[1].flush()
        print("Evaluation done!")
