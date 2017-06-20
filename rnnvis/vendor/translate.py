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

from rnnvis.vendor import data_utils
from rnnvis.vendor.data_utils import read_data
from rnnvis.vendor.seq2seq_model import Seq2SeqModel, create_model

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
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("from_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("from_dev_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_dev_data", None, "Training data.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 400,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
# tf.app.flags.DEFINE_boolean("use_fp16", False,
#                             "Train using fp16 instead of fp32.")
tf.app.flags.DEFINE_boolean("use_lstm", True, "use LSTM.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
# _buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
bucket = (30, 30)


def config_proto():
    return tf.ConfigProto(device_count={"GPU": 1}, allow_soft_placement=True)


def train():
    """Train a en->fr translation model using WMT data."""
    from_train = None
    from_dev = None
    if FLAGS.from_train_data:
        from_train_data = FLAGS.from_train_data
        from_dev_data = from_train_data
        if FLAGS.from_dev_data:
            from_dev_data = FLAGS.from_dev_data
        from_train, from_dev, from_vocab_path = prepare_data(
            FLAGS.data_dir,
            from_train_data,
            from_dev_data,
            FLAGS.from_vocab_size)
    else:
        # Prepare WMT data.
        print("Preparing WMT data in %s" % FLAGS.data_dir)
        from_train, from_dev, from_vocab_path = data_utils.prepare_wmt_data(
            FLAGS.data_dir, FLAGS.from_vocab_size)

    vocab = data_utils.initialize_vocabulary(from_vocab_path)
    with tf.Session(config=config_proto()) as sess:
        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, vocab, vocab, bucket, FLAGS, False)

        # Read data into buckets and compute their sizes.
        print("Reading development and training data (limit: %d)."
              % FLAGS.max_train_data_size)
        dev_set = read_data(from_dev)
        train_set = read_data(from_train, FLAGS.max_train_data_size)
        # train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        # train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        # train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
        #                        for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            # random_number_01 = np.random.random_sample()
            # bucket_id = min([i for i in xrange(len(train_buckets_scale))
            #                  if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_set)
            _, step_loss, _, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print("global step %d learning rate %.4f step-time %.2f perplexity "
                      "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
                before_save(checkpoint_path)
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                # for bucket_id in xrange(len(_buckets)):
                if len(dev_set) == 0:
                    print("  eval: empty bucket")
                    continue
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set)
                _, eval_loss, _, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                             target_weights, True)
                eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
                    "inf")
                print("  eval: bucket perplexity %.2f" % eval_ppx)
                sys.stdout.flush()


def decode():
    with tf.Session(config=config_proto()) as sess:
        # Load vocabularies.
        en_vocab_path = os.path.join(FLAGS.data_dir,
                                     "vocab%d.from" % FLAGS.from_vocab_size)
        en_vocab, rev_en_vocab = data_utils.initialize_vocabulary(en_vocab_path)

        # Create model and load parameters.
        model = create_model(sess, en_vocab, en_vocab, bucket, FLAGS, True)
        model.batch_size = 1  # We decode one sentence at a time.

        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            # Get token-ids for the input sentence.
            token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
            # Which bucket does it belong to?
            # bucket_id = len(_buckets) - 1
            # for i, bucket in enumerate(_buckets):
            #     if bucket[0] >= len(token_ids):
            #         bucket_id = i
            #         break
            if bucket[1] < len(token_ids):
                logging.warning("Sentence truncated: %s", sentence)

            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                [(token_ids, [])])
            # Get output logits for the sentence.
            _, _, output_logits, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                             target_weights, True)
            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            # If there is an EOS symbol in outputs, cut them at that point.
            if data_utils.EOS_ID in outputs:
                outputs = outputs[:outputs.index(data_utils.EOS_ID)]
            # Print out French sentence corresponding to outputs.
            print(" ".join([tf.compat.as_str(rev_en_vocab[output]) for output in outputs]))
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()


def self_test():
    """Test the translation model."""
    with tf.Session(config=config_proto()) as sess:
        print("Self-test for neural translation model.")
        # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
        model = Seq2SeqModel(10, 10, bucket, 32, 2,
                             5.0, 32, 0.3, 0.99, num_samples=8, use_lstm=FLAGS.use_lstm)
        sess.run(tf.global_variables_initializer())

        # Fake data set for both the (3, 3) and (6, 6) bucket.
        data_set = [([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6]),
                    ([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])]
        for _ in range(5):  # Train the fake model for 5 steps.
            # bucket_id = random.choice([0, 1])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                data_set)
            model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                       False)


def main(_):
    if FLAGS.self_test:
        self_test()
    elif FLAGS.decode:
        decode()
    else:
        init_tf_environ(0)
        train()


if __name__ == "__main__":
    tf.app.run()
