# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sequence-to-sequence model with an attention mechanism."""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import random
import copy

import numpy as np
import tensorflow as tf
import tensorflow.contrib.legacy_seq2seq as seq2seq

from rnnvis.rnn.base_model import ModelBase
from rnnvis.rnn.seq2seq_overwrite import embedding_rnn_seq2seq
from rnnvis.vendor import data_utils
from rnnvis.rnn.command_utils import data_type


def create_model(session, from_vocab, to_vocab, bucket, options, forward_only):
    """Create translation model and initialize or load parameters in session."""
    dtype = data_type()
    model = Seq2SeqModel(
        from_vocab,
        to_vocab,
        bucket,
        options.size,
        options.num_layers,
        options.max_gradient_norm,
        options.batch_size,
        options.learning_rate,
        options.learning_rate_decay_factor,
        use_lstm=options.use_lstm,
        forward_only=forward_only,
        dtype=dtype)
    ckpt = tf.train.get_checkpoint_state(options.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
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
                 size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 use_lstm=False,
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
        def build_vocab(vocab):
            id_to_word = [None] * len(vocab)
            for key, value in vocab.items():
                id_to_word[value] = key.decode('utf-8')
            word_to_id = {word: i for i, word in enumerate(id_to_word)}
            return word_to_id, id_to_word
        self.source_vocab, self.rev_source_vocab = build_vocab(source_vocab)
        self.target_vocab, self.rev_target_vocab = build_vocab(target_vocab)
        self.bucket = bucket
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
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
                    dtype)

            softmax_loss_function = sampled_loss

        # Create the internal multi-layer cell for our RNN.
        def single_cell():
            return tf.contrib.rnn.GRUCell(size)

        if use_lstm:
            def single_cell():
                return tf.contrib.rnn.BasicLSTMCell(size)
        cell = single_cell()
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return embedding_rnn_seq2seq(
                encoder_inputs,
                decoder_inputs,
                cell,
                num_encoder_symbols=len(source_vocab),
                num_decoder_symbols=len(target_vocab),
                embedding_size=size,
                output_projection=output_projection,
                feed_previous=do_decode,
                dtype=dtype)

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in range(bucket[0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}".format(i)))
        for i in range(bucket[1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(dtype, shape=[None],
                                                      name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1]
                   for i in range(len(self.decoder_inputs) - 1)]

        # Training outputs and losses.
        # if forward_only:
        self.outputs, self.encoder_states, self.decoder_states = \
            seq2seq_f(self.encoder_inputs[:bucket[0]], self.decoder_inputs[:bucket[1]], forward_only)
        self.loss = seq2seq.sequence_loss(self.outputs, targets[:bucket[1]], self.target_weights[:bucket[1]],
                                          softmax_loss_function=softmax_loss_function)
        # If we use output projection, we need to project outputs for decoding.
        if forward_only and (output_projection is not None):
            # for b in range(len(bucket)):
            self.outputs = [
                tf.matmul(output, output_projection[0]) + output_projection[1]
                for output in self.outputs
                ]
        # else:
        #     self.outputs, self.encoder_states, self.decoder_states = \
        #         seq2seq_f(self.encoder_inputs[:bucket[0]], self.decoder_inputs[:bucket[1]], False)
        #     self.loss = seq2seq.sequence_loss(self.outputs, targets[:bucket[1]], self.target_weights[:bucket[1]],
        #                                       softmax_loss_function=softmax_loss_function)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not forward_only:
            # self.gradient_norms = []
            # self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            # for b in range(len(buckets)):
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                             max_gradient_norm)
            self.gradient_norm = norm
            self.update = opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables())

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

    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
             forward_only):
        """Run a step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
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

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None, None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:(decoder_size+1)], outputs[(decoder_size+1):]
            # No gradient norm, loss, outputs.

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
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                                  [data_utils.PAD_ID] * decoder_pad_size)

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
                if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights

    def get_batches(self, data):
        for i in range(0, len(data)-self.batch_size+1, self.batch_size):
            yield self.get_batch(data[i:(i+self.batch_size)], sample=False)

    def evaluate_and_record(self, sess, inputs, targets, recorder, verbose=True, refresh_state=False):
        pass
