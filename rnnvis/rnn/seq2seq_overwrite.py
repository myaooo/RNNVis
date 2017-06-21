import collections
import copy

import tensorflow as tf

from rnnvis.rnn.constant import EmbeddingWrapper, RNNCell, OutputProjectionWrapper


def is_sequence(x):
    return isinstance(x, collections.Sequence)


def _yield_flat_nest(nest):
    for n in nest:
        if is_sequence(n):
            for ni in _yield_flat_nest(n):
                yield ni
        else:
            yield n


def flatten(nest):
    """Returns a flat sequence from a given nested structure.

    If `nest` is not a sequence, this returns a single-element list: `[nest]`.

    Args:
      nest: an arbitrarily nested structure or a scalar object.
        Note, numpy arrays are considered scalars.

    Returns:
      A Python list, the flattened version of the input.
    """
    return list(_yield_flat_nest(nest)) if is_sequence(nest) else [nest]


def static_rnn_with_states(cell, inputs, initial_state=None, dtype=None,
                           scope=None):
    """Creates a recurrent neural network specified by RNNCell `cell`.

    The simplest form of RNN network generated is:

    ```python
      state = cell.zero_state(...)
      outputs = []
      for input_ in inputs:
        output, state = cell(input_, state)
        outputs.append(output)
      return (outputs, state)
    ```
    However, a few other options are available:

    An initial state can be provided.
    If the sequence_length vector is provided, dynamic calculation is performed.
    This method of calculation does not compute the RNN steps past the maximum
    sequence length of the minibatch (thus saving computational time),
    and properly propagates the state at an example's sequence length
    to the final state output.

    The dynamic calculation performed is, at time `t` for batch row `b`,

    ```python
      (output, state)(b, t) =
        (t >= sequence_length(b))
          ? (zeros(cell.output_size), states(b, sequence_length(b) - 1))
          : cell(input(b, t), state(b, t - 1))
    ```

    Args:
      cell: An instance of RNNCell.
      inputs: A length T list of inputs, each a `Tensor` of shape
        `[batch_size, input_size]`, or a nested tuple of such elements.
      initial_state: (optional) An initial state for the RNN.
        If `cell.state_size` is an integer, this must be
        a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
        If `cell.state_size` is a tuple, this should be a tuple of
        tensors having shapes `[batch_size, s] for s in cell.state_size`.
      dtype: (optional) The data type for the initial state and expected output.
        Required if initial_state is not provided or RNN state has a heterogeneous
        dtype.
      scope: VariableScope for the created subgraph; defaults to "rnn".

    Returns:
      A pair (outputs, state) where:

      - outputs is a length T list of outputs (one for each input), or a nested
        tuple of such elements.
      - state is the final state

    Raises:
      TypeError: If `cell` is not an instance of RNNCell.
      ValueError: If `inputs` is `None` or an empty list, or if the input depth
        (column size) cannot be inferred from inputs via shape inference.
    """

    if not isinstance(cell, RNNCell):
        raise TypeError("cell must be an instance of RNNCell")
    if not is_sequence(inputs):
        raise TypeError("inputs must be a sequence")
    if not inputs:
        raise ValueError("inputs must not be empty")

    outputs = []
    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    with tf.variable_scope(scope or "rnn") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        # Obtain the first sequence of the input
        first_input = inputs
        while is_sequence(first_input):
            first_input = first_input[0]

        # Temporarily avoid EmbeddingWrapper and seq2seq badness
        # TODO(lukaszkaiser): remove EmbeddingWrapper
        if first_input.get_shape().ndims != 1:

            input_shape = first_input.get_shape().with_rank_at_least(2)
            fixed_batch_size = input_shape[0]

            flat_inputs = flatten(inputs)
            for flat_input in flat_inputs:
                input_shape = flat_input.get_shape().with_rank_at_least(2)
                batch_size, input_size = input_shape[0], input_shape[1:]
                fixed_batch_size.merge_with(batch_size)
                for i, size in enumerate(input_size):
                    if size.value is None:
                        raise ValueError(
                            "Input size (dimension %d of inputs) must be accessible via "
                            "shape inference, but saw value None." % i)
        else:
            fixed_batch_size = first_input.get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
        else:
            batch_size = tf.shape(first_input)[0]
        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError("If no initial_state is provided, "
                                 "dtype must be specified")
            state = cell.zero_state(batch_size, dtype)

        states = []
        for time, input_ in enumerate(inputs):
            if time > 0:
                varscope.reuse_variables()
            # pylint: disable=cell-var-from-loop
            call_cell = lambda: cell(input_, state)
            # pylint: enable=cell-var-from-loop
            output, state = call_cell()

            outputs.append(output)
            states.append(state)

        return outputs, states


def rnn_decoder(decoder_inputs,
                initial_state,
                cell,
                loop_function=None,
                scope=None):
    """RNN decoder for the sequence-to-sequence model.

    Args:
      decoder_inputs: A list of 2D Tensors [batch_size x input_size].
      initial_state: 2D Tensor with shape [batch_size x cell.state_size].
      cell: core_rnn_cell.RNNCell defining the cell function and size.
      loop_function: If not None, this function will be applied to the i-th output
        in order to generate the i+1-st input, and decoder_inputs will be ignored,
        except for the first element ("GO" symbol). This can be used for decoding,
        but also for training to emulate http://arxiv.org/abs/1506.03099.
        Signature -- loop_function(prev, i) = next
          * prev is a 2D Tensor of shape [batch_size x output_size],
          * i is an integer, the step number (when advanced control is needed),
          * next is a 2D Tensor of shape [batch_size x input_size].
      scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
          shape [batch_size x output_size] containing generated outputs.
        state: The state of each cell at the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].
          (Note that in some cases, like basic RNN cell or GRU cell, outputs and
           states can be the same. They are different for LSTM cells though.)
    """
    with tf.variable_scope(scope or "rnn_decoder"):
        state = initial_state
        outputs = []
        states = []
        prev = None
        for i, inp in enumerate(decoder_inputs):
            if loop_function is not None and prev is not None:
                with tf.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            output, state = cell(inp, state)
            outputs.append(output)
            states.append(state)
            if loop_function is not None:
                prev = output
    return outputs, states


def _extract_argmax_and_embed(embedding,
                              output_projection=None,
                              update_embedding=True):
    """Get a loop_function that extracts the previous symbol and embeds it.

    Args:
      embedding: embedding tensor for symbols.
      output_projection: None or a pair (W, B). If provided, each fed previous
        output will first be multiplied by W and added B.
      update_embedding: Boolean; if False, the gradients will not propagate
        through the embeddings.

    Returns:
      A loop function.
    """

    def loop_function(prev, _):
        if output_projection is not None:
            prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
        prev_symbol = tf.argmax(prev, 1)
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = tf.stop_gradient(emb_prev)
        return emb_prev

    return loop_function


def embedding_rnn_decoder(decoder_inputs,
                          initial_state,
                          cell,
                          num_symbols,
                          embedding_size,
                          output_projection=None,
                          feed_previous=False,
                          update_embedding_for_previous=True,
                          scope=None):
    """RNN decoder with embedding and a pure-decoding option.

    Args:
      decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
      initial_state: 2D Tensor [batch_size x cell.state_size].
      cell: core_rnn_cell.RNNCell defining the cell function.
      num_symbols: Integer, how many symbols come into the embedding.
      embedding_size: Integer, the length of the embedding vector for each symbol.
      output_projection: None or a pair (W, B) of output projection weights and
        biases; W has shape [output_size x num_symbols] and B has
        shape [num_symbols]; if provided and feed_previous=True, each fed
        previous output will first be multiplied by W and added B.
      feed_previous: Boolean; if True, only the first of decoder_inputs will be
        used (the "GO" symbol), and all other decoder inputs will be generated by:
          next = embedding_lookup(embedding, argmax(previous_output)),
        In effect, this implements a greedy decoder. It can also be used
        during training to emulate http://arxiv.org/abs/1506.03099.
        If False, decoder_inputs are used as given (the standard decoder case).
      update_embedding_for_previous: Boolean; if False and feed_previous=True,
        only the embedding for the first symbol of decoder_inputs (the "GO"
        symbol) will be updated by back propagation. Embeddings for the symbols
        generated from the decoder itself remain unchanged. This parameter has
        no effect if feed_previous=False.
      scope: VariableScope for the created subgraph; defaults to
        "embedding_rnn_decoder".

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors. The
          output is of shape [batch_size x cell.output_size] when
          output_projection is not None (and represents the dense representation
          of predicted tokens). It is of shape [batch_size x num_decoder_symbols]
          when output_projection is None.
        state: The state of each decoder cell in each time-step. This is a list
          with length len(decoder_inputs) -- one item for each time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].

    Raises:
      ValueError: When output_projection has the wrong shape.
    """
    with tf.variable_scope(scope or "embedding_rnn_decoder") as scope:
        if output_projection is not None:
            dtype = scope.dtype
            proj_weights = tf.convert_to_tensor(output_projection[0], dtype=dtype)
            proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
            proj_biases = tf.convert_to_tensor(output_projection[1], dtype=dtype)
            proj_biases.get_shape().assert_is_compatible_with([num_symbols])

        embedding = tf.get_variable("embedding",
                                    [num_symbols, embedding_size])
        loop_function = _extract_argmax_and_embed(
            embedding, output_projection,
            update_embedding_for_previous) if feed_previous else None
        emb_inp = (tf.nn.embedding_lookup(embedding, i)
                   for i in decoder_inputs)
        return rnn_decoder(
            emb_inp, initial_state, cell, loop_function=loop_function)


def embedding_rnn_seq2seq(encoder_inputs,
                          decoder_inputs,
                          cell,
                          num_encoder_symbols,
                          num_decoder_symbols,
                          embedding_size,
                          output_projection=None,
                          feed_previous=False,
                          dtype=None,
                          scope=None):
    """Embedding RNN sequence-to-sequence model.

    This model first embeds encoder_inputs by a newly created embedding (of shape
    [num_encoder_symbols x input_size]). Then it runs an RNN to encode
    embedded encoder_inputs into a state vector. Next, it embeds decoder_inputs
    by another newly created embedding (of shape [num_decoder_symbols x
    input_size]). Then it runs RNN decoder, initialized with the last
    encoder state, on embedded decoder_inputs.

    Args:
      encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
      decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
      cell: core_rnn_cell.RNNCell defining the cell function and size.
      num_encoder_symbols: Integer; number of symbols on the encoder side.
      num_decoder_symbols: Integer; number of symbols on the decoder side.
      embedding_size: Integer, the length of the embedding vector for each symbol.
      output_projection: None or a pair (W, B) of output projection weights and
        biases; W has shape [output_size x num_decoder_symbols] and B has
        shape [num_decoder_symbols]; if provided and feed_previous=True, each
        fed previous output will first be multiplied by W and added B.
      feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
        of decoder_inputs will be used (the "GO" symbol), and all other decoder
        inputs will be taken from previous outputs (as in embedding_rnn_decoder).
        If False, decoder_inputs are used as given (the standard decoder case).
      dtype: The dtype of the initial state for both the encoder and encoder
        rnn cells (default: tf.float32).
      scope: VariableScope for the created subgraph; defaults to
        "embedding_rnn_seq2seq"

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors. The
          output is of shape [batch_size x cell.output_size] when
          output_projection is not None (and represents the dense representation
          of predicted tokens). It is of shape [batch_size x num_decoder_symbols]
          when output_projection is None.
        state: The state of each decoder cell in each time-step. This is a list
          with length len(decoder_inputs) -- one item for each time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].
    """
    with tf.variable_scope(scope or "embedding_rnn_seq2seq") as scope:
        if dtype is not None:
            scope.set_dtype(dtype)
        else:
            dtype = scope.dtype

        # Encoder.
        encoder_cell = copy.deepcopy(cell)
        encoder_cell = EmbeddingWrapper(
            encoder_cell,
            embedding_classes=num_encoder_symbols,
            embedding_size=embedding_size)
        _, encoder_states = static_rnn_with_states(
            encoder_cell, encoder_inputs, dtype=dtype)

        # Decoder.
        if output_projection is None:
            cell = OutputProjectionWrapper(cell, num_decoder_symbols)

        # if isinstance(feed_previous, bool):
        outputs, states = embedding_rnn_decoder(
            decoder_inputs,
            encoder_states[-1],
            cell,
            num_decoder_symbols,
            embedding_size,
            output_projection=output_projection,
            feed_previous=feed_previous)
        return outputs, states, encoder_states
