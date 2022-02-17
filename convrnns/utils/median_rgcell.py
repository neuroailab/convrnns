import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
from convrnns.utils import model_tool
from convrnns.utils.cell_utils import memory, harbor, _ds_conv, residual_add
from convrnns.utils.cells import ConvRNNCell
from convrnns.utils.main import _get_func_from_kwargs
import six


class ReciprocalGateCell(ConvRNNCell):
    """
    memory with cell and output that both, by default, perfectly integrate incoming information;
    the cell then gates the input to the output, and the output gates the input to the cell.
    both cell and output are pseudo-residual recurrent networks, as well.
    """

    def __init__(
        self,
        shape,
        out_depth,
        cell_depth,
        tau_filter_size,
        gate_filter_size,
        ff_filter_size,
        in_out_filter_size=[3, 3],
        cell_tau_filter_size=None,
        feedback_filter_size=[3, 3],
        feedback_entry="out",
        feedback_depth_separable=False,
        ff_depth_separable=False,
        in_out_depth_separable=False,
        gate_depth_separable=False,
        tau_depth_separable=False,
        tau_nonlinearity=tf.tanh,
        gate_nonlinearity=tf.tanh,
        tau_bias=0.0,
        gate_bias=0.0,
        tau_multiplier=-1.0,
        gate_multiplier=-1.0,
        tau_offset=1.0,
        gate_offset=1.0,
        input_activation=tf.identity,
        feedback_activation=tf.identity,
        cell_activation=tf.nn.elu,
        out_activation=tf.nn.elu,
        cell_residual=False,
        out_residual=False,
        residual_to_cell_tau=False,
        residual_to_cell_gate=False,
        residual_to_out_tau=False,
        residual_to_out_gate=False,
        input_to_tau=False,
        input_to_gate=False,
        input_to_cell=False,
        input_to_out=False,
        cell_to_out=False,
        kernel_initializer="xavier",
        kernel_initializer_kwargs=None,
        bias_initializer=tf.zeros_initializer,
        weight_decay=None,
        layer_norm=False,
        norm_gain=1.0,
        norm_shift=0.0,
    ):
        """
        Initialize the memory function of the ReciprocalGateCell.

        Args:
            #TODO

        """

        self.shape = shape  # [H, W]

        def ksize(val):
            if isinstance(val, float):
                return [int(val), int(val)]
            elif isinstance(val, int):
                return [val, val]
            else:
                return val

        self.tau_filter_size = ksize(tau_filter_size)
        self.gate_filter_size = ksize(gate_filter_size)
        self.cell_tau_filter_size = ksize(cell_tau_filter_size)
        self.ff_filter_size = ksize(ff_filter_size)
        self.feedback_filter_size = ksize(feedback_filter_size)
        self.in_out_filter_size = ksize(in_out_filter_size)

        if cell_tau_filter_size is not None:
            self.cell_tau_filter_size = cell_tau_filter_size
        else:
            self.cell_tau_filter_size = self.tau_filter_size

        self.tau_depth_separable = tau_depth_separable
        self.ff_depth_separable = ff_depth_separable
        self.gate_depth_separable = gate_depth_separable
        self.in_out_depth_separable = in_out_depth_separable

        if self.gate_filter_size == [0, 0]:
            self.use_cell = False
        else:
            self.use_cell = True

        self.feedback_entry = feedback_entry
        self.feedback_depth_separable = feedback_depth_separable
        self.cell_depth = cell_depth
        self.out_depth = out_depth
        self.cell_residual = cell_residual
        self.out_residual = out_residual
        self.residual_to_cell_tau = residual_to_cell_tau
        self.residual_to_cell_gate = residual_to_cell_gate
        self.residual_to_out_tau = residual_to_out_tau
        self.residual_to_out_gate = residual_to_out_gate
        self.input_to_tau = input_to_tau
        self.input_to_gate = input_to_gate
        self.input_to_cell = input_to_cell
        self.input_to_out = input_to_out
        self.cell_to_out = cell_to_out

        if isinstance(gate_nonlinearity, six.string_types):
            self._gate_nonlinearity = _get_func_from_kwargs(gate_nonlinearity)[0]
        else:
            self._gate_nonlinearity = gate_nonlinearity

        if isinstance(tau_nonlinearity, six.string_types):
            self._tau_nonlinearity = _get_func_from_kwargs(tau_nonlinearity)[0]
        else:
            self._tau_nonlinearity = tau_nonlinearity

        self._tau_bias = tau_bias
        self._gate_bias = gate_bias
        self._tau_offset = tau_offset
        self._gate_offset = gate_offset
        self._tau_k = tau_multiplier
        self._gate_k = gate_multiplier

        self._size = tf.TensorShape([self.shape[0], self.shape[1], self.out_depth])
        self._cell_size = tf.TensorShape(
            [self.shape[0], self.shape[1], self.cell_depth]
        )

        if isinstance(feedback_activation, six.string_types):
            self._feedback_activation = _get_func_from_kwargs(feedback_activation)[0]
        else:
            self._feedback_activation = feedback_activation

        if isinstance(input_activation, six.string_types):
            self._input_activation = _get_func_from_kwargs(input_activation)[0]
        else:
            self._input_activation = input_activation

        if isinstance(cell_activation, six.string_types):
            self._cell_activation = _get_func_from_kwargs(cell_activation)[0]
        else:
            self._cell_activation = cell_activation

        if self._cell_activation.__name__ == "crelu":
            print("using crelu! doubling cell out depth")
            self.cell_depth_out = 2 * self.cell_depth
        else:
            self.cell_depth_out = self.cell_depth

        if isinstance(out_activation, six.string_types):
            self._out_activation = _get_func_from_kwargs(out_activation)[0]
        else:
            self._out_activation = out_activation

        if kernel_initializer_kwargs is None:
            kernel_initializer_kwargs = {}

        self._kernel_initializer = model_tool.initializer(
            kind=kernel_initializer, **kernel_initializer_kwargs
        )
        self._bias_initializer = bias_initializer

        if weight_decay is None:
            weight_decay = 0.0
        self._weight_decay = weight_decay
        self._layer_norm = layer_norm
        self._g = norm_gain
        self._b = norm_shift

    def state_size(self):
        return {"cell": self._cell_size, "out": self._size}

    def output_size(self):
        return self._size

    def zero_state(self, batch_size, dtype):
        """
        Return zero-filled state tensor(s)
        """
        shape = self.shape

        out_depth = self.out_depth
        out_zeros = tf.zeros([batch_size, shape[0], shape[1], out_depth], dtype=dtype)

        if self.use_cell:
            cell_depth = self.cell_depth
            cell_zeros = tf.zeros(
                [batch_size, shape[0], shape[1], cell_depth], dtype=dtype
            )
            return tf.concat(values=[cell_zeros, out_zeros], axis=3, name="zero_state")
        else:
            return tf.identity(out_zeros, name="zero_state")

    def _norm(self, inp, scope, dtype=tf.float32):
        shape = inp.get_shape()[-1:]
        gamma_init = tf.constant_initializer(self._g)
        beta_init = tf.constant_initializer(self._b)
        with tf.variable_scope(scope):
            gamma = tf.get_variable(
                shape=shape, initializer=gamma_init, name="gamma", dtype=dtype
            )
            beta = tf.get_variable(
                shape=shape, initializer=beta_init, name="beta", dtype=dtype
            )

        normalized = tf.contrib.layers.layer_norm(inp, reuse=True, scope=scope)
        return normalized

    def __call__(self, inputs, state, fb_input, res_input):
        """
        Produce outputs of RecipCell, given inputs and previous state {'cell':cell_state, 'out':out_state}

        inputs: dict w keys ('ff', 'fb'). ff and fb inputs must have the same shape.
        """

        dtype = inputs.dtype

        if self.use_cell:
            prev_cell, prev_out = tf.split(
                value=state,
                num_or_size_splits=[self.cell_depth, self.out_depth],
                axis=3,
                name="state_split",
            )
        else:
            prev_out = state

        with tf.variable_scope(type(self).__name__):  # "ReciprocalGateCell"

            with tf.variable_scope("input"):

                if self.feedback_entry == "input" and fb_input is not None:
                    if self.feedback_depth_separable:
                        fb_input = _ds_conv(
                            fb_input,
                            self.feedback_filter_size,
                            out_depth=inputs.shape.as_list()[-1],
                            bias=True,
                            scope="feedback",
                            kernel_initializer=self._kernel_initializer,
                            bias_initializer=self._bias_initializer,
                            weight_decay=self._weight_decay,
                        )
                        inputs += self._feedback_activation(fb_input)
                    else:
                        fb_to_in_kernel = tf.get_variable(
                            "feedback_to_input_weights",
                            [
                                self.feedback_filter_size[0],
                                self.feedback_filter_size[1],
                                fb_input.shape.as_list()[-1],
                                inputs.shape.as_list()[-1],
                            ],
                            dtype=dtype,
                            initializer=self._kernel_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(
                                self._weight_decay
                            ),
                        )
                        fb_to_in_bias = tf.get_variable(
                            "feedback_to_input_bias",
                            [inputs.shape.as_list()[-1]],
                            initializer=self._bias_initializer,
                        )

                        inputs += self._feedback_activation(
                            tf.nn.conv2d(
                                fb_input,
                                fb_to_in_kernel,
                                strides=[1, 1, 1, 1],
                                padding="SAME",
                            )
                            + fb_to_in_bias
                        )

                inputs = self._input_activation(inputs, name="inputs")

            if self.use_cell:
                with tf.variable_scope("cell"):

                    # cell tau kernel
                    if not self.tau_depth_separable:
                        cell_to_cell_kernel = tf.get_variable(
                            "cell_to_cell_weights",
                            [
                                self.cell_tau_filter_size[0],
                                self.cell_tau_filter_size[1],
                                self.cell_depth_out,
                                self.cell_depth,
                            ],
                            dtype=dtype,
                            initializer=self._kernel_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(
                                self._weight_decay
                            ),
                        )

                        cell_to_cell_bias = tf.get_variable(
                            "cell_to_cell_bias",
                            [self.cell_depth],
                            initializer=self._bias_initializer,
                        )

                    # gating kernel
                    if not self.gate_depth_separable:
                        out_to_cell_kernel = tf.get_variable(
                            "out_to_cell_weights",
                            [
                                self.gate_filter_size[0],
                                self.gate_filter_size[1],
                                self.out_depth,
                                self.cell_depth,
                            ],
                            dtype=dtype,
                            initializer=self._kernel_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(
                                self._weight_decay
                            ),
                        )

                        out_to_cell_bias = tf.get_variable(
                            "out_to_cell_bias",
                            [self.cell_depth],
                            initializer=self._bias_initializer,
                        )

                    # if cell depth and out depth are different, need to change channel number of input
                    cell_input = tf.zeros_like(
                        prev_cell, dtype=tf.float32, name="cell_input"
                    )
                    assert self.cell_residual or self.input_to_cell
                    if self.cell_residual:
                        assert res_input is not None
                    if res_input is not None and self.cell_residual:
                        if self.ff_depth_separable:
                            cell_input += _ds_conv(
                                res_input,
                                self.ff_filter_size,
                                out_depth=self.cell_depth,
                                bias=True,
                                scope="res_to_cell",
                                kernel_initializer=self._kernel_initializer,
                                bias_initializer=self._bias_initializer,
                                weight_decay=self._weight_decay,
                            )
                        else:
                            res_to_cell_kernel = tf.get_variable(
                                "residual_to_cell_weights",
                                [
                                    self.ff_filter_size[0],
                                    self.ff_filter_size[1],
                                    res_input.shape.as_list()[-1],
                                    self.cell_depth,
                                ],
                                dtype=dtype,
                                initializer=self._kernel_initializer,
                                regularizer=tf.contrib.layers.l2_regularizer(
                                    self._weight_decay
                                ),
                            )

                            res_to_cell_bias = tf.get_variable(
                                "residual_to_cell_bias",
                                [self.cell_depth],
                                initializer=self._bias_initializer,
                            )

                            cell_input += (
                                tf.nn.conv2d(
                                    res_input,
                                    res_to_cell_kernel,
                                    strides=[1, 1, 1, 1],
                                    padding="SAME",
                                )
                                + res_to_cell_bias
                            )
                    if self.input_to_cell:
                        if self.ff_depth_separable:
                            cell_input += _ds_conv(
                                inputs,
                                self.ff_filter_size,
                                out_depth=self.cell_depth,
                                bias=True,
                                scope="input_to_cell",
                                kernel_initializer=self._kernel_initializer,
                                bias_initializer=self._bias_initializer,
                                weight_decay=self._weight_decay,
                            )
                        else:
                            in_to_cell_kernel = tf.get_variable(
                                "input_to_cell_weights",
                                [
                                    self.ff_filter_size[0],
                                    self.ff_filter_size[1],
                                    self.out_depth,
                                    self.cell_depth,
                                ],
                                dtype=dtype,
                                initializer=self._kernel_initializer,
                                regularizer=tf.contrib.layers.l2_regularizer(
                                    self._weight_decay
                                ),
                            )

                            in_to_cell_bias = tf.get_variable(
                                "input_to_cell_bias",
                                [self.cell_depth],
                                initializer=self._bias_initializer,
                            )

                            cell_input += (
                                tf.nn.conv2d(
                                    inputs,
                                    in_to_cell_kernel,
                                    strides=[1, 1, 1, 1],
                                    padding="SAME",
                                )
                                + in_to_cell_bias
                            )

                    if fb_input is not None and self.feedback_entry == "cell":
                        if self.feedback_depth_separable:
                            fb_input = _ds_conv(
                                fb_input,
                                self.feedback_filter_size,
                                out_depth=self.cell_depth,
                                bias=True,
                                scope="feedback",
                                kernel_initializer=self._kernel_initializer,
                                bias_initializer=self._bias_initializer,
                                weight_decay=self._weight_decay,
                            )
                            cell_input += self._feedback_activation(fb_input)
                        else:
                            feedback_to_cell_kernel = tf.get_variable(
                                "feedback_to_cell_weights",
                                [
                                    self.feedback_filter_size[0],
                                    self.feedback_filter_size[1],
                                    fb_input.shape.as_list()[-1],
                                    self.cell_depth,
                                ],
                                dtype=dtype,
                                initializer=self._kernel_initializer,
                                regularizer=tf.contrib.layers.l2_regularizer(
                                    self._weight_decay
                                ),
                            )
                            feedback_to_cell_bias = tf.get_variable(
                                "feedback_to_cell_bias",
                                [self.cell_depth],
                                initializer=self._bias_initializer,
                            )

                            if (
                                fb_input.shape.as_list()[1:3]
                                == prev_cell.shape.as_list()[1:3]
                            ):
                                cell_input += self._feedback_activation(
                                    tf.nn.conv2d(
                                        fb_input,
                                        feedback_to_cell_kernel,
                                        strides=[1, 1, 1, 1],
                                        padding="SAME",
                                    )
                                    + feedback_to_cell_bias
                                )
                            else:
                                assert (
                                    self.feedback_filter_size
                                    == fb_input.shape.as_list()[1:3]
                                )  # only allow fully connected case for now
                                cell_input += self._feedback_activation(
                                    tf.nn.conv2d(
                                        fb_input,
                                        feedback_to_cell_kernel,
                                        strides=[1, 1, 1, 1],
                                        padding="VALID",
                                    )
                                    + feedback_to_cell_bias
                                )
                    # ops
                    if self.tau_depth_separable:
                        cell_tau = _ds_conv(
                            prev_cell,
                            self.cell_tau_filter_size,
                            bias=True,
                            scope="tau",
                            kernel_initializer=self._kernel_initializer,
                            bias_initializer=self._bias_initializer,
                            weight_decay=self._weight_decay,
                        )
                    else:
                        cell_tau = (
                            tf.nn.conv2d(
                                prev_cell,
                                cell_to_cell_kernel,
                                strides=[1, 1, 1, 1],
                                padding="SAME",
                                name="cell_tau",
                            )
                            + cell_to_cell_bias
                            + self._tau_bias
                        )
                    # cell gate
                    if self.gate_depth_separable:
                        cell_gate = _ds_conv(
                            prev_out,
                            self.gate_filter_size,
                            out_depth=self.cell_depth,
                            bias=True,
                            scope="out_to_cell",
                            kernel_initializer=self._kernel_initializer,
                            bias_initializer=self._bias_initializer,
                            weight_decay=self._weight_decay,
                        )
                    else:
                        cell_gate = (
                            tf.nn.conv2d(
                                prev_out,
                                out_to_cell_kernel,
                                strides=[1, 1, 1, 1],
                                padding="SAME",
                                name="cell_gate",
                            )
                            + out_to_cell_bias
                            + self._gate_bias
                        )
                    if self.input_to_tau:
                        input_to_cell_tau_kernel = tf.get_variable(
                            "input_to_cell_tau_weights",
                            [
                                self.cell_tau_filter_size[0],
                                self.cell_tau_filter_size[1],
                                self.out_depth,
                                self.cell_depth,
                            ],
                            dtype=dtype,
                            initializer=self._kernel_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(
                                self._weight_decay
                            ),
                        )
                        cell_tau += tf.nn.conv2d(
                            inputs,
                            input_to_cell_tau_kernel,
                            strides=[1, 1, 1, 1],
                            padding="SAME",
                        )
                    if res_input is not None and self.residual_to_cell_tau:
                        res_to_cell_tau_kernel = tf.get_variable(
                            "residual_to_cell_tau_weights",
                            [
                                self.cell_tau_filter_size[0],
                                self.cell_tau_filter_size[1],
                                res_input.shape.as_list()[-1],
                                self.cell_depth,
                            ],
                            dtype=dtype,
                            initializer=self._kernel_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(
                                self._weight_decay
                            ),
                        )
                        # print("res input to tau", res_input.name, res_input.shape)
                        cell_tau += tf.nn.conv2d(
                            res_input,
                            res_to_cell_tau_kernel,
                            strides=[1, 1, 1, 1],
                            padding="SAME",
                        )

                    if self.input_to_gate:
                        input_to_cell_gate_kernel = tf.get_variable(
                            "input_to_cell_gate_weights",
                            [
                                self.gate_filter_size[0],
                                self.gate_filter_size[1],
                                self.out_depth,
                                self.cell_depth,
                            ],
                            dtype=dtype,
                            initializer=self._kernel_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(
                                self._weight_decay
                            ),
                        )
                        cell_gate += tf.nn.conv2d(
                            inputs,
                            input_to_cell_gate_kernel,
                            strides=[1, 1, 1, 1],
                            padding="SAME",
                        )
                    if res_input is not None and self.residual_to_cell_gate:
                        res_to_cell_gate_kernel = tf.get_variable(
                            "residual_to_cell_gate_weights",
                            [
                                self.gate_filter_size[0],
                                self.gate_filter_size[1],
                                res_input.shape.as_list()[-1],
                                self.cell_depth,
                            ],
                            dtype=dtype,
                            initializer=self._kernel_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(
                                self._weight_decay
                            ),
                        )
                        # print("res input to gate", res_input.name, res_input.shape)
                        cell_gate += tf.nn.conv2d(
                            res_input,
                            res_to_cell_gate_kernel,
                            strides=[1, 1, 1, 1],
                            padding="SAME",
                        )

                    cell_tau = self._tau_nonlinearity(cell_tau)
                    cell_gate = self._gate_nonlinearity(cell_gate)

                    next_cell = (
                        self._tau_offset + self._tau_k * cell_tau
                    ) * prev_cell + (
                        self._gate_offset + self._gate_k * cell_gate
                    ) * cell_input

                    if self._layer_norm:
                        next_cell = self._norm(
                            next_cell, scope="layer_norm", dtype=dtype
                        )
                    next_cell = self._cell_activation(next_cell)

            with tf.variable_scope("out"):

                # out tau kernel
                if not self.tau_depth_separable:
                    out_to_out_kernel = tf.get_variable(
                        "out_to_out_weights",
                        [
                            self.tau_filter_size[0],
                            self.tau_filter_size[1],
                            self.out_depth,
                            self.out_depth,
                        ],
                        dtype=dtype,
                        initializer=self._kernel_initializer,
                        regularizer=tf.contrib.layers.l2_regularizer(
                            self._weight_decay
                        ),
                    )

                    out_to_out_bias = tf.get_variable(
                        "out_to_out_bias",
                        [self.out_depth],
                        initializer=self._bias_initializer,
                    )

                # out gate kernelt
                if self.use_cell:
                    if not self.gate_depth_separable:
                        cell_to_out_kernel = tf.get_variable(
                            "cell_to_out_weights",
                            [
                                self.gate_filter_size[0],
                                self.gate_filter_size[1],
                                self.cell_depth_out,
                                self.out_depth,
                            ],
                            dtype=dtype,
                            initializer=self._kernel_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(
                                self._weight_decay
                            ),
                        )

                        cell_to_out_bias = tf.get_variable(
                            "cell_to_out_bias",
                            [self.out_depth],
                            initializer=self._bias_initializer,
                        )

                if self.input_to_out:
                    if self.in_out_depth_separable:
                        out_input = _ds_conv(
                            inputs,
                            self.in_out_filter_size,
                            out_depth=self.out_depth,
                            bias=True,
                            scope="input_to_out",
                            kernel_initializer=self._kernel_initializer,
                            bias_initializer=self._bias_initializer,
                            weight_decay=self._weight_decay,
                        )
                    else:
                        in_to_out_kernel = tf.get_variable(
                            "input_to_out_weights",
                            [
                                self.in_out_filter_size[0],
                                self.in_out_filter_size[1],
                                self.out_depth,
                                self.out_depth,
                            ],
                            dtype=dtype,
                            initializer=self._kernel_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(
                                self._weight_decay
                            ),
                        )

                        in_to_out_bias = tf.get_variable(
                            "input_to_out_bias",
                            [self.out_depth],
                            initializer=self._bias_initializer,
                        )

                        out_input = (
                            tf.nn.conv2d(
                                inputs,
                                in_to_out_kernel,
                                strides=[1, 1, 1, 1],
                                padding="SAME",
                                name="out_input",
                            )
                            + in_to_out_bias
                        )
                else:
                    out_input = tf.identity(inputs, name="out_input")

                if self.cell_to_out and self.use_cell:

                    if self.gate_depth_separable:
                        out_input += _ds_conv(
                            prev_cell,
                            self.gate_filter_size,
                            out_depth=self.out_depth,
                            bias=True,
                            scope="cell_to_out",
                            kernel_initializer=self._kernel_initializer,
                            bias_initializer=self._bias_initializer,
                            weight_decay=self._weight_decay,
                        )
                    else:
                        out_input += (
                            tf.nn.conv2d(
                                prev_cell,
                                cell_to_out_kernel,
                                strides=[1, 1, 1, 1],
                                padding="SAME",
                                name="out_gate",
                            )
                            + cell_to_out_bias
                            + self._gate_bias
                        )

                if res_input is not None and self.out_residual:
                    out_input = residual_add(out_input, res_input)

                if fb_input is not None and self.feedback_entry == "out":
                    if self.feedback_depth_separable:
                        fb_input = _ds_conv(
                            fb_input,
                            self.feedback_filter_size,
                            out_depth=self.out_depth,
                            bias=True,
                            scope="feedback",
                            kernel_initializer=self._kernel_initializer,
                            bias_initializer=self._bias_initializer,
                            weight_decay=self._weight_decay,
                        )
                        out_input += self._feedback_activation(fb_input)
                    else:
                        feedback_to_out_kernel = tf.get_variable(
                            "feedback_to_out_weights",
                            [
                                self.feedback_filter_size[0],
                                self.feedback_filter_size[1],
                                fb_input.shape.as_list()[-1],
                                self.out_depth,
                            ],
                            dtype=dtype,
                            initializer=self._kernel_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(
                                self._weight_decay
                            ),
                        )
                        feedback_to_out_bias = tf.get_variable(
                            "feedback_to_out_bias",
                            [self.out_depth],
                            initializer=self._bias_initializer,
                        )

                        if (
                            fb_input.shape.as_list()[1:3]
                            == prev_out.shape.as_list()[1:3]
                        ):
                            out_input += self._feedback_activation(
                                tf.nn.conv2d(
                                    fb_input,
                                    feedback_to_out_kernel,
                                    strides=[1, 1, 1, 1],
                                    padding="SAME",
                                )
                                + feedback_to_out_bias
                            )
                        else:
                            assert (
                                self.feedback_filter_size
                                == fb_input.shape.as_list()[1:3]
                            )  # only allow fully connected case for now
                            out_input += self._feedback_activation(
                                tf.nn.conv2d(
                                    fb_input,
                                    feedback_to_out_kernel,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                )
                                + feedback_to_out_bias
                            )
                # ops
                if self.tau_depth_separable:
                    out_tau = _ds_conv(
                        prev_out,
                        self.tau_filter_size,
                        bias=True,
                        scope="tau",
                        kernel_initializer=self._kernel_initializer,
                        bias_initializer=self._bias_initializer,
                        weight_decay=self._weight_decay,
                    )
                else:
                    out_tau = (
                        tf.nn.conv2d(
                            prev_out,
                            out_to_out_kernel,
                            strides=[1, 1, 1, 1],
                            padding="SAME",
                            name="out_tau",
                        )
                        + out_to_out_bias
                        + self._tau_bias
                    )

                if self.use_cell and not self.cell_to_out:
                    if self.gate_depth_separable:
                        out_gate = _ds_conv(
                            prev_cell,
                            self.gate_filter_size,
                            out_depth=self.out_depth,
                            bias=True,
                            scope="gate",
                            kernel_initializer=self._kernel_initializer,
                            bias_initializer=self._bias_initializer,
                            weight_decay=self._weight_decay,
                        )
                    else:
                        out_gate = (
                            tf.nn.conv2d(
                                prev_cell,
                                cell_to_out_kernel,
                                strides=[1, 1, 1, 1],
                                padding="SAME",
                                name="out_gate",
                            )
                            + cell_to_out_bias
                            + self._gate_bias
                        )
                else:
                    out_gate = tf.zeros(
                        shape=out_input.shape.as_list(),
                        dtype=tf.float32,
                        name="out_gate",
                    )

                if self.input_to_tau:
                    input_to_out_tau_kernel = tf.get_variable(
                        "input_to_out_tau_weights",
                        [
                            self.tau_filter_size[0],
                            self.tau_filter_size[1],
                            self.out_depth,
                            self.out_depth,
                        ],
                        dtype=dtype,
                        initializer=self._kernel_initializer,
                        regularizer=tf.contrib.layers.l2_regularizer(
                            self._weight_decay
                        ),
                    )
                    out_tau += tf.nn.conv2d(
                        inputs,
                        input_to_out_tau_kernel,
                        strides=[1, 1, 1, 1],
                        padding="SAME",
                    )
                if res_input is not None and self.residual_to_out_tau:
                    res_to_out_tau_kernel = tf.get_variable(
                        "residual_to_out_tau_weights",
                        [
                            self.tau_filter_size[0],
                            self.tau_filter_size[1],
                            res_input.shape.as_list()[-1],
                            self.out_depth,
                        ],
                        dtype=dtype,
                        initializer=self._kernel_initializer,
                        regularizer=tf.contrib.layers.l2_regularizer(
                            self._weight_decay
                        ),
                    )
                    out_tau += tf.nn.conv2d(
                        res_input,
                        res_to_out_tau_kernel,
                        strides=[1, 1, 1, 1],
                        padding="SAME",
                    )

                if self.input_to_gate:
                    input_to_out_gate_kernel = tf.get_variable(
                        "input_to_out_gate_weights",
                        [
                            self.gate_filter_size[0],
                            self.gate_filter_size[1],
                            self.out_depth,
                            self.out_depth,
                        ],
                        dtype=dtype,
                        initializer=self._kernel_initializer,
                        regularizer=tf.contrib.layers.l2_regularizer(
                            self._weight_decay
                        ),
                    )
                    out_gate += tf.nn.conv2d(
                        inputs,
                        input_to_out_gate_kernel,
                        strides=[1, 1, 1, 1],
                        padding="SAME",
                    )
                if res_input is not None and self.residual_to_out_gate:
                    if self.gate_depth_separable:
                        out_gate += _ds_conv(
                            res_input,
                            self.gate_filter_size,
                            out_depth=self.out_depth,
                            bias=False,
                            scope="residual_to_out_gate",
                            kernel_initializer=self._kernel_initializer,
                            bias_initializer=self._bias_initializer,
                            weight_decay=self._weight_decay,
                        )
                    else:
                        res_to_out_gate_kernel = tf.get_variable(
                            "residual_to_out_gate_weights",
                            [
                                self.gate_filter_size[0],
                                self.gate_filter_size[1],
                                res_input.shape.as_list()[-1],
                                self.out_depth,
                            ],
                            dtype=dtype,
                            initializer=self._kernel_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(
                                self._weight_decay
                            ),
                        )
                        out_gate += tf.nn.conv2d(
                            res_input,
                            res_to_out_gate_kernel,
                            strides=[1, 1, 1, 1],
                            padding="SAME",
                        )

                out_tau = self._tau_nonlinearity(out_tau)
                out_gate = self._gate_nonlinearity(out_gate)

                next_out = (self._tau_offset + self._tau_k * out_tau) * prev_out + (
                    self._gate_offset + self._gate_k * out_gate
                ) * out_input

                if self._layer_norm:
                    next_out = self._norm(next_out, scope="layer_norm", dtype=dtype)
                next_out = self._out_activation(next_out)

                if self.use_cell:
                    next_state = tf.concat(axis=3, values=[next_cell, next_out])
                else:
                    next_state = next_out

            return next_out, next_state


class tnn_median_ReciprocalGateCell(ConvRNNCell):
    def __init__(
        self,
        harbor_shape,
        harbor=(harbor, None),
        pre_memory=None,
        memory=(memory, None),
        post_memory=None,
        input_init=(tf.zeros, None),
        state_init=(tf.zeros, None),
        dtype=tf.float32,
        name=None,
    ):

        self.harbor_shape = harbor_shape
        self.harbor = harbor if harbor[1] is not None else (harbor[0], {})
        self.pre_memory = pre_memory
        self.memory = memory if memory[1] is not None else (memory[0], {})
        self.post_memory = post_memory

        self.input_init = (
            input_init if input_init[1] is not None else (input_init[0], {})
        )
        self.state_init = (
            state_init if state_init[1] is not None else (state_init[0], {})
        )

        self.dtype_tmp = dtype
        self.name_tmp = name

        self._reuse = None

        # signature: Res3Cell(shape, ff_filter_size, cell_filter_size, cell_depth, out_depth, **kwargs)
        self._strides = self.pre_memory[0][1].get("strides", [1, 1, 1, 1])[1:3]
        self.memory[1]["shape"] = self.memory[1].get(
            "shape",
            [
                self.harbor_shape[1] // self._strides[0],
                self.harbor_shape[2] // self._strides[1],
            ],
        )

        idx = [
            i
            for i in range(len(self.pre_memory))
            if "out_depth" in self.pre_memory[i][1]
        ][0]
        self._pre_conv_idx = idx
        if "out_depth" not in self.memory[1]:
            self.memory[1]["out_depth"] = self.pre_memory[idx][1]["out_depth"]

        self.conv_cell = ReciprocalGateCell(**self.memory[1])

    def __call__(self, inputs=None, state=None):
        """
        Produce outputs given inputs
        If inputs or state are None, they are initialized from scratch.
        :Kwargs:
            - inputs (list)
                A list of inputs. Inputs are combined using the harbor function
            - state
        :Returns:
            (output, state)
        """

        with tf.variable_scope(self.name_tmp, reuse=self._reuse):

            if inputs is None:
                inputs = [
                    self.input_init[0](shape=self.harbor_shape, **self.input_init[1])
                ]

            # separate feedback from feedforward input
            fb_input = None
            if len(inputs) == 1:
                ff_idx = 0
                output = self.harbor[0](
                    inputs,
                    self.harbor_shape,
                    self.name_tmp,
                    reuse=self._reuse,
                    **self.harbor[1]
                )
            elif len(inputs) > 1:
                for j, inp in enumerate(inputs):
                    if self.pre_memory[self._pre_conv_idx][1]["input_name"] in inp.name:
                        ff_inpnm = inp.name
                        ff_idx = j
                        ff_depth = inputs[ff_idx].shape.as_list()[-1]
                output = self.harbor[0](
                    inputs,
                    self.harbor_shape,
                    self.name_tmp,
                    ff_inpnm=ff_inpnm,
                    reuse=self._reuse,
                    **self.harbor[1]
                )
                fb_depth = output.shape.as_list()[-1] - ff_depth
                if self.harbor[1]["channel_op"] == "concat":
                    output, fb_input = tf.split(
                        output, num_or_size_splits=[ff_depth, fb_depth], axis=3
                    )

            res_input = None
            pre_name_counter = 0
            for function, kwargs in self.pre_memory:
                with tf.variable_scope(
                    "pre_" + str(pre_name_counter), reuse=self._reuse
                ):
                    if function.__name__ == "component_conv":
                        if kwargs.get("return_input", False):
                            output, res_input = function(
                                output, [inputs[ff_idx]], **kwargs
                            )  # component_conv needs to know the inputs
                        else:
                            output = function(
                                output, [inputs[ff_idx]], **kwargs
                            )  # component_conv needs to know the inputs

                    else:
                        output = function(output, **kwargs)
                pre_name_counter += 1

            if state is None:
                batch_size = output.get_shape().as_list()[0]
                state = self.conv_cell.zero_state(batch_size, dtype=self.dtype_tmp)

            output, state = self.conv_cell(output, state, fb_input, res_input)
            self.state = tf.identity(state, name="state")

            post_name_counter = 0
            for function, kwargs in self.post_memory:
                with tf.variable_scope(
                    "post_" + str(post_name_counter), reuse=self._reuse
                ):
                    if function.__name__ == "component_conv":
                        output = function(output, inputs, **kwargs)
                    else:
                        output = function(output, **kwargs)
                post_name_counter += 1

            self.output_tmp = tf.identity(
                tf.cast(output, self.dtype_tmp), name="output"
            )

            # Now reuse variables across time
            self._reuse = True

        self.state_shape = self.conv_cell.state_size()  # DELETE?
        self.output_tmp_shape = self.output_tmp.shape  # DELETE?
        return self.output_tmp, self.state

    @property
    def state_size(self):
        """
        Size(s) of state(s) used by this cell.
        It can be represented by an Integer, a TensorShape or a tuple of Integers
        or TensorShapes.
        """
        # if self.state is not None:
        return self.state_shape
        # else:
        #     raise ValueError('State not initialized yet')

    @property
    def output_size(self):
        """
        Integer or TensorShape: size of outputs produced by this cell.
        """
        # if self.output_tmp is not None:
        return self.output_tmp_shape
        # else:
        #     raise ValueError('Output not initialized yet')


def _conv(
    inp,
    filter_size,
    out_depth,
    bias,
    scope,
    bias_initializer=None,
    kernel_initializer=None,
    bias_regularizer=None,
    kernel_regularizer=None,
    data_format="channels_last",
):
    """convolution:
    Args:
      args: a 4D Tensor or a list of 4D, batch x n, Tensors.
      filter_size: int tuple of filter height and width.
      out_depth: int, number of features.
      bias: boolean as to whether to have a bias.
      bias_initializer: starting value to initialize the bias.
      kernel_initializer: starting value to initialize the kernel.
    Returns:
      A 4D Tensor with shape [batch h w out_depth]
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """

    # Calculate the total size of arguments on dimension 1.

    dtype = inp.dtype
    shape = inp.shape.as_list()
    if data_format == "channels_last":
        h = shape[1]
        w = shape[2]
        in_depth = shape[3]
        data_format = "NHWC"
    elif data_format == "channels_first":
        h = shape[2]
        w = shape[3]
        in_depth = shape[1]
        data_format = "NCHW"

    if filter_size[0] > h:
        filter_size[0] = h
    if filter_size[1] > w:
        filter_size[1] = w

    if kernel_regularizer is None:
        kernel_regularizer = 0.0
    if bias_regularizer is None:
        bias_regularizer = 0.0
    if kernel_initializer is None:
        kernel_initializer = tf.contrib.layers.xavier_initializer()
    if bias_initializer is None:
        bias_initializer = tf.contrib.layers.xavier_initializer()

    # Now the computation.
    with tf.variable_scope(scope):
        kernel = tf.get_variable(
            "weights",
            [filter_size[0], filter_size[1], in_depth, out_depth],
            dtype=dtype,
            initializer=kernel_initializer,
            regularizer=tf.contrib.layers.l2_regularizer(kernel_regularizer),
        )

        out = tf.nn.conv2d(
            inp, kernel, strides=[1, 1, 1, 1], padding="SAME", data_format=data_format
        )
        if not bias:
            return out
        if bias_initializer is None:
            bias_initializer = tf.constant_initializer(0.0, dtype=dtype)
        bias_term = tf.get_variable(
            "bias",
            [out_depth],
            dtype=dtype,
            initializer=bias_initializer,
            regularizer=tf.contrib.layers.l2_regularizer(bias_regularizer),
        )
        return out + bias_term
