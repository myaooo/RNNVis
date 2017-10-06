import random
import os
import time
import math
import sys
import yaml

import numpy as np
import tensorflow as tf
import tensorflow.contrib.legacy_seq2seq as seq2seq

from rnnvis.rnn.base_model import ModelBase
from rnnvis.rnn.seq2seq_overwrite import embedding_rnn_seq2seq
from rnnvis.rnn import seq2seq_utils
from rnnvis.rnn.command_utils import data_type, config_proto
from rnnvis.rnn.config_utils import get_rnn_cell
from rnnvis.utils.io_utils import before_save, get_path
from rnnvis.db import get_dataset


def dict2obj(d):
    class Struct(object):
        def __init__(self, **entries):
            self.__dict__.update(entries)
    return Struct(**d)


def build_model(config_file, forward_only=True):
    if isinstance(config_file, str):
        with open(config_file) as f:
            config_dict = yaml.safe_load(f)
    elif isinstance(config_file, dict):
        config_dict = config_file
    else:
        raise ValueError("Unsupported type {:s} for arg:config_file!".format(type(config_file)))
    model_config = config_dict['model']
    if model_config.get('app', None) != 'seq2seq':
        return None, None
    train_config = config_dict['train']
    dtype = data_type()
    vocab, rev_vocab = get_dataset(model_config['dataset'], ['word_to_id'],
                                   vocab_size=model_config['vocab_size']
                                   )['word_to_id']
    size = model_config['cells'][0]['num_units']
    num_layers = len(model_config['cells'])
    max_grad_norm = train_config.get('gradient_clip_args', {}).get('clip_norm', 5.0)
    model = Seq2SeqModel(
        vocab, vocab,
        model_config['bucket'],
        model_config['name'],
        size, num_layers,
        max_grad_norm,
        train_config['batch_size'],
        train_config['learning_rate'],
        train_config.get('learning_rate_decay_factor', 0.95),
        cell=model_config['cell_type'],
        forward_only=forward_only,
        dtype=dtype
    )
    # model_config = dict2obj(model_config)
    train_config['dataset'] = model_config['dataset']
    train_config = dict2obj(train_config)
    return model, train_config


def create_model(from_vocab, to_vocab, bucket, options, forward_only):
    """Create translation model and initialize or load parameters in session."""
    dtype = data_type()
    model = Seq2SeqModel(
        from_vocab,
        to_vocab,
        bucket,
        options.name,
        options.size,
        options.num_layers,
        options.max_gradient_norm,
        options.batch_size,
        options.learning_rate,
        options.learning_rate_decay_factor,
        cell='BasicLSTM' if options.use_lstm else 'GRU',
        forward_only=forward_only,
        dtype=dtype)
    ckpt = tf.train.get_checkpoint_state(model.logdir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.restore()
    return model


class Seq2SeqModel(ModelBase):
    """Sequence-to-sequence model with attention and for multiple buckets.

    This class implements a multi-layer recurrent neural network as encoder,
    and an attention-based decoder. This is the same as the model described in
    this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
    or into the seq2seq library for complete model implementation.
    This class also allows to use GRU cells in addition to LSTM cells, and
    sampled softmax to handle large output vocabulary size. A single-layer
    version of this model, but with bi-directional encoder, was presented in
      http://arxiv.org/abs/1409.0473
    and sampled softmax is described in Section 3 of the following paper.
      http://arxiv.org/abs/1412.2007
    """

    def __init__(self,
                 source_vocab,
                 target_vocab,
                 bucket,
                 name,
                 size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 cell='GRU',
                 num_samples=512,
                 forward_only=False,
                 dtype=tf.float32):
        """Create the model.

        Args:
          source_vocab: size of the source vocabulary.
          target_vocab: size of the target vocabulary.
          buckets: a list of pairs (I, O), where I specifies maximum input length
            that will be processed in that bucket, and O specifies maximum output
            length. Training instances that have inputs longer than I or outputs
            longer than O will be pushed to the next bucket and padded accordingly.
            We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
          size: number of units in each layer of the model.
          num_layers: number of layers in the model.
          max_gradient_norm: gradients will be clipped to maximally this norm.
          batch_size: the size of the batches used during training;
            the model construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g., for decoding.
          learning_rate: learning rate to start with.
          learning_rate_decay_factor: decay learning rate by this much when needed.
          use_lstm: if true, we use LSTM cells instead of GRU cells.
          num_samples: number of samples for sampled softmax.
          forward_only: if set, we do not construct the backward pass in the model.
          dtype: the data type to use to store internal variables.
        """
        self.graph = tf.Graph()

        def build_vocab(vocab):
            id_to_word = [None] * len(vocab)
            for key, value in vocab.items():
                id_to_word[value] = key.decode('utf-8')
            word_to_id = {word: i for i, word in enumerate(id_to_word)}
            return word_to_id, id_to_word
        self.name = name
        self.size = size
        self.num_layers = num_layers
        self.cell = get_rnn_cell(cell)
        self.dtype = data_type()
        self.source_vocab, self.rev_source_vocab = build_vocab(source_vocab)
        self.target_vocab, self.rev_target_vocab = build_vocab(target_vocab)
        self.bucket = bucket
        self.batch_size = batch_size
        self.forward_only = forward_only
        self.max_gradient_norm = max_gradient_norm
        self._build_model(learning_rate, learning_rate_decay_factor, num_samples)

    def _build_model(self, learning_rate, learning_rate_decay_factor, num_samples):
        with self.graph.as_default():
            self.learning_rate = tf.Variable(
                float(learning_rate), trainable=False, dtype=self.dtype)
            self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)
            self.global_step = tf.Variable(0, trainable=False)

            # If we use sampled softmax, we need an output projection.
            output_projection = None
            softmax_loss_function = None
            # Sampled softmax only makes sense if we sample less than vocabulary size.
            if num_samples > 0 and num_samples < self.target_vocab_size:
                w_t = tf.get_variable("proj_w", [self.target_vocab_size, self.size], dtype=self.dtype)
                w = tf.transpose(w_t)
                b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=self.dtype)
                output_projection = (w, b)

                def sampled_loss(labels, logits):
                    labels = tf.reshape(labels, [-1, 1])
                    # We need to compute the sampled_softmax_loss using 32bit floats to
                    # avoid numerical instabilities.
                    local_w_t = tf.cast(w_t, tf.float32)
                    local_b = tf.cast(b, tf.float32)
                    local_inputs = tf.cast(logits, tf.float32)
                    return tf.cast(
                        tf.nn.sampled_softmax_loss(
                            weights=local_w_t,
                            biases=local_b,
                            labels=labels,
                            inputs=local_inputs,
                            num_sampled=num_samples,
                            num_classes=self.target_vocab_size),
                        self.dtype)

                softmax_loss_function = sampled_loss

            # Create the internal multi-layer cell for our RNN.
            def single_cell():
                return self.cell(self.size)

            cell = single_cell()
            if self.num_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(self.num_layers)])

            # The seq2seq function: we use embedding for the input and attention.
            def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
                return embedding_rnn_seq2seq(
                    encoder_inputs,
                    decoder_inputs,
                    cell,
                    num_encoder_symbols=len(self.source_vocab),
                    num_decoder_symbols=len(self.target_vocab),
                    embedding_size=self.size,
                    output_projection=output_projection,
                    feed_previous=do_decode,
                    dtype=self.dtype)

            # Feeds for inputs.
            self.encoder_inputs = []
            self.decoder_inputs = []
            self.target_weights = []
            for i in range(self.encoder_size):  # Last bucket is the biggest one.
                self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                          name="encoder{0}".format(i)))
            for i in range(self.decoder_size + 1):
                self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                          name="decoder{0}".format(i)))
                self.target_weights.append(tf.placeholder(self.dtype, shape=[None],
                                                          name="weight{0}".format(i)))

            # Our targets are decoder inputs shifted by one.
            targets = [self.decoder_inputs[i + 1]
                       for i in range(len(self.decoder_inputs) - 1)]

            # Training outputs and losses.
            self.outputs, self.encoder_states, self.decoder_states = \
                seq2seq_f(self.encoder_inputs[:self.encoder_size],
                          self.decoder_inputs[:self.decoder_size],
                          self.forward_only)
            self.loss = seq2seq.sequence_loss(self.outputs,
                                              targets[:self.decoder_size],
                                              self.target_weights[:self.decoder_size],
                                              softmax_loss_function=softmax_loss_function)
            # If we use output projection, we need to project outputs for decoding.
            if self.forward_only and (output_projection is not None):
                # for b in range(len(bucket)):
                self.outputs = [
                    tf.matmul(output, output_projection[0]) + output_projection[1]
                    for output in self.outputs
                    ]
            # Gradients and SGD update operation for training the model.
            params = tf.trainable_variables()
            if not self.forward_only:
                # self.gradient_norms = []
                # self.updates = []
                opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                # for b in range(len(buckets)):
                gradients = tf.gradients(self.loss, params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                 self.max_gradient_norm)
                self.gradient_norm = norm
                self.update = opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step)

            self.finalized = False
            self._sess = None
            self.logdir = get_path('./models', self.name)
            self.finalize()

    @property
    def encoder_size(self):
        return self.bucket[0]

    @property
    def decoder_size(self):
        return self.bucket[1]

    @property
    def source_vocab_size(self):
        return len(self.source_vocab)

    @property
    def target_vocab_size(self):
        return len(self.target_vocab)

    def get_word_from_id(self, ids, target=False):
        vocab = self.rev_target_vocab if target else self.rev_source_vocab
        if isinstance(ids, int):
            ids = [ids]
        words = [vocab[i] for i in ids if 0 <= i < len(vocab)]
        return words

    def get_id_from_word(self, words, target=False):
        vocab = self.target_vocab if target else self.source_vocab
        if isinstance(words, str):
            words = [words]
        ids = [vocab.get(i, seq2seq_utils.UNK_ID) for i in words]
        return ids

    def run_with_context(self, func, *args, **kwargs):
        self.finalize()
        with self.graph.as_default():
            return func(self.sess, *args, **kwargs)

    def save(self, path=None, step=None):
        """
        Save the model to a given path
        :param path:
        :return:
        """
        path = path if path is not None else os.path.join(self.logdir, 'model')
        before_save(path)
        self.saver.save(self.sess, path, global_step=self.global_step)
        print("Model variables saved to {}.".format(get_path(path, absolute=True)))

    def restore(self, path=None):
        path = path if path is not None else self.logdir
        checkpoint = tf.train.latest_checkpoint(path)
        self.saver.restore(self.sess, checkpoint)
        print("Model variables restored from {}.".format(get_path(path, absolute=True)))

    def finalize(self):
        """
        After all the computation ops are built in the graph, build a supervisor which implicitly finalize the graph
        :return: None
        """
        if self.finalized:
            # print("Graph has already been finalized!")
            return False
        with self.graph.as_default():
            self._init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver(tf.trainable_variables())
        self.finalized = True
        return True

    @property
    def sess(self):
        assert self.finalized
        if self._sess is None or self._sess._closed:
            self._sess = tf.Session(graph=self.graph, config=config_proto())
            self._sess.run(self._init_op)
        return self._sess
        # return self.supervisor.managed_session(config=config_proto())

    def step(self, encoder_inputs, decoder_inputs, target_weights,
             forward_only):
        """Run a step of the model feeding the given inputs.

        Args:
          encoder_inputs: list of numpy int vectors to feed as encoder inputs.
          decoder_inputs: list of numpy int vectors to feed as decoder inputs.
          target_weights: list of numpy float vectors to feed as target weights.
          forward_only: whether to do the backward step or only forward.

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.

        Raises:
          ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """
        # Check if the sizes match.
        encoder_size, decoder_size = self.bucket
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.update,  # Update Op that does SGD.
                           self.gradient_norm,  # Gradient norm.
                           self.loss]  # Loss for this batch.
        else:
            output_feed = [self.loss]  # Loss for this batch.
            for l in range(decoder_size):  # Output logits.
                output_feed.append(self.outputs[l])
            for l in range(encoder_size):
                output_feed.append(self.encoder_states[l])
            for l in range(decoder_size):
                output_feed.append(self.decoder_states[l])

        outputs = self.sess.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None, None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:(decoder_size+1)], outputs[(decoder_size+1):]
            # No gradient norm, loss, outputs.

    def train(self, train_set, dev_set, steps_per_checkpoint=400):

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        with self.sess as sess:
            while True:
                # Get a batch and make a step.
                start_time = time.time()
                encoder_inputs, decoder_inputs, target_weights = self.get_batch(
                    train_set)
                _, step_loss, _, _ = self.step(encoder_inputs, decoder_inputs,
                                                target_weights, False)
                step_time += (time.time() - start_time) / steps_per_checkpoint
                loss += step_loss / steps_per_checkpoint
                current_step += 1

                # Once in a while, we save checkpoint, print statistics, and run evals.
                if current_step % steps_per_checkpoint == 0:
                    # Print statistics for the previous epoch.
                    perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                    print("global step %d learning rate %.4f step-time %.2f perplexity "
                          "%.2f" % (self.global_step.eval(), self.learning_rate.eval(),
                                    step_time, perplexity))
                    # Decrease learning rate if no improvement was seen over last 3 times.
                    if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                        self.sess.run(self.learning_rate_decay_op)
                    previous_losses.append(loss)
                    # Save checkpoint and zero timer and loss.
                    self.save()
                    step_time, loss = 0.0, 0.0
                    # Run evals on development set and print their perplexity.
                    # for bucket_id in xrange(len(_buckets)):
                    if len(dev_set) == 0:
                        print("  eval: empty bucket")
                        continue
                    encoder_inputs, decoder_inputs, target_weights = self.get_batch(dev_set)
                    _, eval_loss, _, _ = self.step(encoder_inputs, decoder_inputs,
                                                    target_weights, True)
                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
                        "inf")
                    print("  eval: bucket perplexity %.2f" % eval_ppx)
                    sys.stdout.flush()

    def get_batch(self, data, sample=True):
        """Get a random batch of data from the specified bucket, prepare for step.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
          data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a batch.
          sample: boolean, which bucket to get the batch for.

        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights) for
          the constructed batch that has the proper format to call step(...) later.
        """
        encoder_size, decoder_size = self.bucket
        encoder_inputs, decoder_inputs = [], []

        def get_data(data, i):
            return data[i]
        if sample:
            def get_data(data, i):
                return random.choice(data)
        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for i in range(self.batch_size):
            encoder_input, decoder_input = get_data(data, i)

            # Encoder inputs are padded and then reversed.
            encoder_pad = [seq2seq_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([seq2seq_utils.GO_ID] + decoder_input +
                                  [seq2seq_utils.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in range(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in range(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in range(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == seq2seq_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights

    def get_batches(self, data):
        for i in range(0, len(data)-self.batch_size+1, self.batch_size):
            yield self.get_batch(data[i:(i+self.batch_size)], sample=False)

    def evaluate_and_record(self, sess, inputs, targets, recorder, verbose=True, refresh_state=False):
        pass
