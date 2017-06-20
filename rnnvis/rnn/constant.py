import collections

import tensorflow as tf
import tensorflow.contrib.rnn as rnn

RNNCell = rnn.RNNCell
BasicRNNCell = rnn.BasicRNNCell
BasicLSTMCell = rnn.BasicLSTMCell
LSTMCell = rnn.LSTMCell
GRUCell = rnn.GRUCell
MultiRNNCell = rnn.MultiRNNCell
DropOutWrapper = rnn.DropoutWrapper
EmbeddingWrapper = rnn.EmbeddingWrapper
InputProjectionWrapper = rnn.InputProjectionWrapper
OutputProjectionWrapper = rnn.OutputProjectionWrapper
LSTMStateTuple = rnn.LSTMStateTuple
static_rnn = rnn.static_rnn
# rnn.EmbeddingWrapper

tf.GraphKeys.INPUTS = 'my_inputs'


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
      sequence_length: Specifies the length of each sequence in inputs.
        An int32 or int64 vector (tensor) size `[batch_size]`, values in `[0, T)`.
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
            if time > 0: varscope.reuse_variables()
            # pylint: disable=cell-var-from-loop
            call_cell = lambda: cell(input_, state)
            # pylint: enable=cell-var-from-loop
            output, state = call_cell()

            outputs.append(output)
            states.append(state)

        return outputs, states
