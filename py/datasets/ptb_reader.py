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


"""Utilities for parsing PTB text files."""

import os
from .data_utils import *
import tensorflow as tf


def ptb_raw_data(data_path=None):
    """Load PTB raw data from data directory "data_path".
    Reads PTB text files, converts strings to integer ids,
    and performs mini-batching of the inputs.
    The PTB datasets comes from Tomas Mikolov's webpage:
    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
    Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.
    Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
    """

    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")
    paths = [train_path, valid_path, test_path]

    data_list, word_to_id = load_data_as_ids(paths)
    data_list.append(word_to_id)

    return tuple(data_list)
