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

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""

import math
import os
# import random
import sys
import time
import logging

import numpy as np
import tensorflow as tf

from rnnvis.rnn import seq2seq_utils
from rnnvis.rnn.seq2seq_utils import read_data
from rnnvis.rnn.seq2seq import Seq2SeqModel
from rnnvis.procedures import build_model

from rnnvis.procedures import init_tf_environ
from rnnvis.utils.io_utils import before_save
from rnnvis.db.seq2seq import prepare_data

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.98,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("from_vocab_size", 10000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("to_vocab_size", 10000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("name", "translate", "Training directory.")
tf.app.flags.DEFINE_string("config", "./config/model/wmt-seq2seq-gru.yml", "Data directory")
# tf.app.flags.DEFINE_string("from_train_data", None, "Training data.")
# tf.app.flags.DEFINE_string("to_train_data", None, "Training data.")
# tf.app.flags.DEFINE_string("from_dev_data", None, "Training data.")
# tf.app.flags.DEFINE_string("to_dev_data", None, "Training data.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 400,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_lstm", True, "use LSTM.")
tf.app.flags.DEFINE_boolean("evaluate", False, "evaluate and record")

FLAGS = tf.app.flags.FLAGS

bucket = (30, 30)


def config_proto():
    return tf.ConfigProto(device_count={"GPU": 1}, allow_soft_placement=True)


def train():
    """Train a en->fr translation model using WMT data."""
    # Prepare WMT data.
    print("Preparing WMT data in %s" % FLAGS.data_dir)
    from_train, from_dev, from_vocab_path = seq2seq_utils.prepare_wmt_data(
        FLAGS.data_dir, FLAGS.from_vocab_size)

    vocab = seq2seq_utils.initialize_vocabulary(from_vocab_path)
    # with tf.Session(config=config_proto()) as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model, train_config = build_model(FLAGS.config, True)
    ckpt = tf.train.get_checkpoint_state(model.logdir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.restore()
    # Read data into buckets and compute their sizes.
    print("Reading development and training data (limit: %d)."
          % FLAGS.max_train_data_size)
    dev_set = read_data(from_dev)
    train_set = read_data(from_train, FLAGS.max_train_data_size)

    model.train(train_set, dev_set, 100)


def decode():
    # Load vocabularies.
    en_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.from" % FLAGS.from_vocab_size)
    en_vocab, rev_en_vocab = seq2seq_utils.initialize_vocabulary(en_vocab_path)

    # Create model and load parameters.
    model, train_config = build_model(FLAGS.config, False)
    model.batch_size = 1  # We decode one sentence at a time.

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
        # Get token-ids for the input sentence.
        token_ids = seq2seq_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
        if bucket[1] < len(token_ids):
            logging.warning("Sentence truncated: %s", sentence)

        # Get a 1-element batch to feed the sentence to the model.
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            [(token_ids, [])])
        # Get output logits for the sentence.
        _, _, output_logits, _ = model.step(encoder_inputs, decoder_inputs,
                                         target_weights, True)
        # This is a greedy decoder - outputs are just argmaxes of output_logits.
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        # If there is an EOS symbol in outputs, cut them at that point.
        if seq2seq_utils.EOS_ID in outputs:
            outputs = outputs[:outputs.index(seq2seq_utils.EOS_ID)]
        # Print out French sentence corresponding to outputs.
        print(" ".join([tf.compat.as_str(rev_en_vocab[output]) for output in outputs]))
        print("> ", end="")
        sys.stdout.flush()
        sentence = sys.stdin.readline()


def self_test():
    """Test the translation model."""
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = Seq2SeqModel(range(10), range(10), bucket, 32, 2,
                         5.0, 32, 0.3, 0.99, num_samples=8)

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = [([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6]),
                ([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])]
    for _ in range(5):  # Train the fake model for 5 steps.
        # bucket_id = random.choice([0, 1])
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            data_set)
        model.step(encoder_inputs, decoder_inputs, target_weights,
                   False)


def eval_record():
    from rnnvis.rnn.seq2seq_evaluator import Seq2SeqEvaluator, Seq2SeqFeeder
    from rnnvis.rnn.eval_recorder import StateRecorder
    en_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.from" % FLAGS.from_vocab_size)
    en_vocab, rev_en_vocab = seq2seq_utils.initialize_vocabulary(en_vocab_path)

    from_train, from_dev, from_vocab_path = seq2seq_utils.prepare_wmt_data(
        FLAGS.data_dir, FLAGS.from_vocab_size)
    # with tf.Session(config=config_proto()) as sess:
    model, train_config = build_model(FLAGS.config, False)
    train_set = read_data(from_train, 1000)
    # model.batch_size =   # We decode one sentence at a time.
    evaluator = Seq2SeqEvaluator(model, True, True, True)
    recorder_encoder = StateRecorder('wmt','seq2seq', flush_every=500, pad_ids={seq2seq_utils.PAD_ID})
    recorder_decoder = StateRecorder('wmt','seq2seq', flush_every=500, pad_ids={seq2seq_utils.PAD_ID})
    evaluator.evaluate_and_record(train_set, [recorder_encoder, recorder_decoder])

def main(_):
    try:
        init_tf_environ(1)
    except:
        init_tf_environ(0)
    if FLAGS.evaluate:
        eval_record()
    elif FLAGS.self_test:
        self_test()
    elif FLAGS.decode:
        decode()
    else:
        train()


if __name__ == "__main__":
    tf.app.run()
