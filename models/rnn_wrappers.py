import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, ResidualWrapper
from .modules import prenet


class DecoderPrenetWrapper(RNNCell):
    '''Runs RNN inputs through a prenet before sending them to the cell.'''

    def __init__(self, cell, is_training):
        super(DecoderPrenetWrapper, self).__init__()
        self._cell = cell
        self._is_training = is_training

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def call(self, inputs, state):
        prenet_out = prenet(inputs, self._is_training, scope='decoder_prenet')
        return self._cell(prenet_out, state)

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)


class ConcatOutputAndAttentionWrapper(RNNCell):
    '''Concatenates RNN cell output with the attention context vector.

    This is expected to wrap a cell wrapped with an AttentionWrapper constructed with
    attention_layer_size=None and output_attention=False. Such a cell's state will include an
    "attention" field that is the context vector.
    '''

    def __init__(self, cell):
        super(ConcatOutputAndAttentionWrapper, self).__init__()
        self._cell = cell
        self._residual_cells = []

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size + self._cell.state_size.attention

    def call(self, inputs, state):
        output, res_state = self._cell(inputs, state)
        attention_spread = res_state.attention
        for residual_cell in self._residual_cells:
            residual_cell.set_attention(attention_spread)
        return tf.concat([output, res_state.attention], axis=-1), res_state

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

    def add_residual_cell(self, cell):
        self._residual_cells.append(cell)


class ResidualAttensionWrapper(ResidualWrapper):
    def __init__(self, cell):
        super(ResidualAttensionWrapper, self).__init__()
        self._cell = cell
        self._attention = None

    def set_attention(self, attention):
        self._attention = attention

    def call(self, inputs, state):
        inputs_new = tf.concat([inputs, self._attention], axis=-1)
        return self._cell(inputs_new, state)
