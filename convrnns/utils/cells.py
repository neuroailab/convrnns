from __future__ import absolute_import, division, print_function
import copy, six
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple, RNNCell
from convrnns.utils import model_tool
from convrnns.utils.main import _get_func_from_kwargs
from convrnns.utils.cell_utils import (
    harbor,
    memory,
    _conv_linear,
    residual_add,
    component_conv,
    ksize,
)


class GenFuncCell(RNNCell):
    """Time Decay recurrent network cell."""

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

        fc_layers = [
            i
            for i in range(len(self.pre_memory))
            if ((self.pre_memory[i][0]).__name__ == "fc")
        ]
        if len(fc_layers) == 0:
            self._strides = self.pre_memory[0][1].get("strides", [1, 1, 1, 1])[1:3]
            self._shape = self.memory[1].get(
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
            if "out_depth" not in self.memory[1]:
                self.out_depth = self.pre_memory[idx][1]["out_depth"]
            self.state_shape = [self.harbor_shape[0]] + self._shape + [self.out_depth]
        else:  # just an fc layer
            self.out_depth = self.pre_memory[fc_layers[-1]][1]["out_depth"]
            self.state_shape = [self.harbor_shape[0], self.out_depth]

        self.internal_time = 0
        self.max_internal_time = self.memory[1].get("max_internal_time", None)

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

            output = self.harbor[0](
                inputs,
                self.harbor_shape,
                self.name_tmp,
                reuse=self._reuse,
                **self.harbor[1]
            )

            res_input = None
            curr_time_suffix = "t" + str(self.internal_time)
            pre_name_counter = 0
            for function, kwargs in self.pre_memory:
                with tf.variable_scope(
                    "pre_" + str(pre_name_counter), reuse=self._reuse
                ):
                    if kwargs.get("time_sep", False):
                        kwargs[
                            "time_suffix"
                        ] = curr_time_suffix  # used for scoping in the op

                    if function.__name__ == "component_conv":
                        if kwargs.get("return_input", False):
                            output, res_input = function(
                                output, inputs, **kwargs
                            )  # component_conv needs to know the inputs
                        else:
                            output = function(
                                output, inputs, **kwargs
                            )  # component_conv needs to know the inputs
                    else:
                        output = function(output, **kwargs)

                pre_name_counter += 1

            no_state = self.memory[1].get("no_state", False)
            if no_state:
                print("Bypassing state")
                self.state_shape = None
                self.state = None
            else:
                mem_kwargs = copy.deepcopy(self.memory[1])
                mem_kwargs.pop("no_state", None)
                mem_kwargs.pop("max_internal_time", None)

                if state is None:
                    state = self.state_init[0](
                        shape=output.shape, dtype=self.dtype_tmp, **self.state_init[1]
                    )

                state = self.memory[0](output, state, **mem_kwargs)
                self.state = tf.identity(state, name="state")

                self.state_shape = self.state.shape

                output = self.state

            post_name_counter = 0
            for function, kwargs in self.post_memory:
                with tf.variable_scope(
                    "post_" + str(post_name_counter), reuse=self._reuse
                ):
                    if kwargs.get("time_sep", False):
                        kwargs[
                            "time_suffix"
                        ] = curr_time_suffix  # used for scoping in the op

                    if function.__name__ == "component_conv":
                        output = function(output, inputs, **kwargs)
                    elif function.__name__ == "residual_add":
                        output = function(output, res_input, **kwargs)
                    else:
                        output = function(output, **kwargs)
                post_name_counter += 1
            self.output_tmp = tf.identity(
                tf.cast(output, self.dtype_tmp), name="output"
            )
            # scope.reuse_variables()
            self._reuse = True
        self.output_shape_tmp = self.output_tmp.shape

        if (self.max_internal_time is None) or (
            (self.max_internal_time is not None)
            and (self.internal_time < self.max_internal_time)
        ):
            self.internal_time += 1

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
        # if self.output is not None:
        return self.output_shape_tmp
        # else:
        #     raise ValueError('Output not initialized yet')


class ConvRNNCell(object):
    """Abstract object representing an Convolutional RNN cell.
    From: https://github.com/loliverhennigh/Convolutional-LSTM-in-Tensorflow/blob/master/BasicConvLSTMCell.py
    """

    def __call__(self, inputs, state, scope=None):
        """Run this RNN cell on inputs, starting from the given state."""
        raise NotImplementedError("Abstract method")

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell."""
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        raise NotImplementedError("Abstract method")

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).
        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.
        Returns:
          tensor of shape '[batch_size x shape[0] x shape[1] x out_depth]
          filled with zeros
        """
        shape = self.shape
        out_depth = self._out_depth
        zeros = tf.zeros([batch_size, shape[0], shape[1], out_depth], dtype=dtype)
        return zeros


class ConvNormBasicCell(ConvRNNCell):
    """Convolutional version of SimpleRNN recurrent network cell."""

    def __init__(
        self,
        shape,
        filter_size,
        out_depth,
        layer_norm=True,
        kernel_regularizer=5e-4,
        bias_regularizer=5e-4,
        activation=tf.nn.elu,
        kernel_initializer=None,
        bias_initializer=None,
    ):
        """Initialize the Conv Norm Basic cell.
        Args:
          shape: int tuple thats the height and width of the cell
          filter_size: int tuple thats the height and width of the filter
          out_depth: int thats the depth of the cell
          activation: Activation function of the inner states.
        """
        self.shape = shape
        self.filter_size = filter_size
        self._out_depth = out_depth
        self._size = tf.TensorShape([self.shape[0], self.shape[1], self._out_depth])
        self._activation = activation
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._layer_norm = layer_norm

    @property
    def state_size(self):
        return self._size

    @property
    def output_size(self):
        return self._size

    def __call__(self, inputs, state):
        """Basic RNN cell."""
        with tf.variable_scope(type(self).__name__):  # "ConvNormBasicCell"
            if self._activation is not None:
                with tf.variable_scope("s"):
                    s = _conv_linear(
                        [state],
                        self.filter_size,
                        self._out_depth,
                        True,
                        self._bias_initializer,
                        self._kernel_initializer,
                        self._bias_regularizer,
                        self._kernel_regularizer,
                    )

                with tf.variable_scope("i"):
                    i = _conv_linear(
                        [inputs],
                        self.filter_size,
                        self._out_depth,
                        True,
                        self._bias_initializer,
                        self._kernel_initializer,
                        self._bias_regularizer,
                        self._kernel_regularizer,
                    )

                if self._layer_norm:
                    new_state = tf.contrib.layers.layer_norm(
                        i + s,
                        activation_fn=self._activation,
                        reuse=tf.AUTO_REUSE,
                        scope="layer_norm",
                    )
                else:
                    new_state = self._activation(i + s)

            return new_state, new_state


class ConvGRUCell(ConvRNNCell):
    """Conv GRU recurrent network cell."""

    def __init__(
        self,
        shape,
        filter_size,
        out_depth,
        weight_decay=0.0,
        forget_bias=1.0,
        activation=tf.nn.tanh,
        kernel_initializer=None,
        kernel_initializer_kwargs=None,
        bias_initializer=None,
        bias_initializer_kwargs=None,
        layer_norm=False,
        norm_gain=1.0,
        norm_shift=0.0,
    ):
        """Initialize the Conv GRU cell.
        Args:
          shape: int tuple thats the height and width of the cell
          filter_size: int tuple thats the height and width of the filter
          out_depth: int thats the depth of the cell
          activation: Activation function of the inner states.
        """
        self.shape = shape
        self.filter_size = ksize(filter_size)
        self._out_depth = out_depth
        self._size = tf.TensorShape([self.shape[0], self.shape[1], self._out_depth])
        self._activation = activation
        self._kernel_initializer = kernel_initializer
        if kernel_initializer_kwargs is None:
            kernel_initializer_kwargs = {}
        if kernel_initializer is not None:
            self._kernel_initializer = model_tool.initializer(
                kind=kernel_initializer, **kernel_initializer_kwargs
            )

        self._bias_initializer = bias_initializer
        if bias_initializer_kwargs is None:
            bias_initializer_kwargs = {}
        if bias_initializer is not None:
            self._bias_initializer = model_tool.initializer(
                kind=bias_initializer, **bias_initializer_kwargs
            )
        self._layer_norm = layer_norm
        self._weight_decay = weight_decay
        self._g = norm_gain
        self._b = norm_shift
        self._forget_bias = forget_bias

    def _norm(self, inp, scope):
        shape = inp.get_shape()[-1:]
        gamma_init = tf.constant_initializer(self._g)
        beta_init = tf.constant_initializer(self._b)
        with tf.variable_scope(scope):
            gamma = tf.get_variable(shape=shape, initializer=gamma_init, name="gamma")
            beta = tf.get_variable(shape=shape, initializer=beta_init, name="beta")

        normalized = tf.contrib.layers.layer_norm(inp, reuse=True, scope=scope)
        return normalized

    @property
    def state_size(self):
        return self._size

    @property
    def output_size(self):
        return self._size

    def __call__(self, inputs, state):
        """Gated recurrent unit (GRU)."""
        with tf.variable_scope(type(self).__name__):  # "ConvGRUCell"
            with tf.variable_scope("gates"):
                # We start with bias of 1.0 to not reset and not update.
                bias_ones = self._bias_initializer
                if self._bias_initializer is None:
                    dtype = [a.dtype for a in [inputs, state]][0]
                    bias_ones = tf.constant_initializer(1.0, dtype=dtype)

                value = _conv_linear(
                    [inputs, state],
                    self.filter_size,
                    2 * self._out_depth,
                    True,
                    bias_ones,
                    self._kernel_initializer,
                    bias_regularizer=self._weight_decay,
                    kernel_regularizer=self._weight_decay,
                )
                r_pre, u_pre = tf.split(value=value, num_or_size_splits=2, axis=3)
                if self._layer_norm:
                    r_pre = self._norm(r_pre, "r_pre")
                    u_pre = self._norm(u_pre, "u_pre")

            r = tf.nn.sigmoid(r_pre + self._forget_bias)
            u = tf.nn.sigmoid(u_pre)

            with tf.variable_scope("candidates"):
                c_pre = _conv_linear(
                    [inputs, r * state],
                    self.filter_size,
                    self._out_depth,
                    True,
                    self._bias_initializer,
                    self._kernel_initializer,
                    bias_regularizer=self._weight_decay,
                    kernel_regularizer=self._weight_decay,
                )

                if self._layer_norm:
                    c_pre = self._norm(c_pre, "c_pre")

                c = self._activation(c_pre)

            new_h = u * state + (1 - u) * c
            return new_h, new_h


class ConvLSTMCell(ConvRNNCell):
    """Conv LSTM recurrent network cell."""

    def __init__(
        self,
        shape,
        filter_size,
        out_depth,
        weight_decay=0.0,
        use_peepholes=False,
        forget_bias=1.0,
        state_is_tuple=False,
        activation=tf.nn.tanh,
        kernel_initializer=None,
        kernel_initializer_kwargs=None,
        bias_initializer=None,
        bias_initializer_kwargs=None,
        layer_norm=False,
        norm_gain=1.0,
        norm_shift=0.0,
    ):
        """Initialize the Conv LSTM cell.
        Args:
          shape: int tuple thats the height and width of the cell
          filter_size: int tuple thats the height and width of the filter
          out_depth: int thats the depth of the cell
          use_peepholes: bool, set True to enable peephole connections
          activation: Activation function of the inner states.
          forget_bias: float, The bias added to forget gates (see above).
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
        """
        self.shape = shape
        self.filter_size = ksize(filter_size)
        self._use_peepholes = use_peepholes
        self._out_depth = out_depth
        self._size = tf.TensorShape([self.shape[0], self.shape[1], self._out_depth])
        self._concat_size = tf.TensorShape(
            [self.shape[0], self.shape[1], 2 * self._out_depth]
        )
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self._kernel_initializer = kernel_initializer
        if kernel_initializer_kwargs is None:
            kernel_initializer_kwargs = {}
        if kernel_initializer is not None:
            self._kernel_initializer = model_tool.initializer(
                kind=kernel_initializer, **kernel_initializer_kwargs
            )

        self._bias_initializer = bias_initializer
        if bias_initializer_kwargs is None:
            bias_initializer_kwargs = {}
        if bias_initializer is not None:
            self._bias_initializer = model_tool.initializer(
                kind=bias_initializer, **bias_initializer_kwargs
            )

        self._layer_norm = layer_norm
        self._weight_decay = weight_decay
        self._g = norm_gain
        self._b = norm_shift

    @property
    def state_size(self):
        return (
            LSTMStateTuple(self._size, self._size)
            if self._state_is_tuple
            else self._concat_size
        )

    @property
    def output_size(self):
        return self._size

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).
        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.
        Returns:
          tensor of shape '[batch_size x shape[0] x shape[1] x out_depth]
          filled with zeros
        """
        # last dimension is replaced by 2 * out_depth = (c, h)
        shape = self.shape
        out_depth = self._out_depth
        if self._state_is_tuple:
            zeros = LSTMStateTuple(
                tf.zeros([batch_size, shape[0], shape[1], out_depth], dtype=dtype),
                tf.zeros([batch_size, shape[0], shape[1], out_depth], dtype=dtype),
            )
        else:
            zeros = tf.zeros(
                [batch_size, shape[0], shape[1], out_depth * 2], dtype=dtype
            )
        return zeros

    def _norm(self, inp, scope):
        shape = inp.get_shape()[-1:]
        gamma_init = tf.constant_initializer(self._g)
        beta_init = tf.constant_initializer(self._b)
        with tf.variable_scope(scope):
            gamma = tf.get_variable(shape=shape, initializer=gamma_init, name="gamma")
            beta = tf.get_variable(shape=shape, initializer=beta_init, name="beta")

        normalized = tf.contrib.layers.layer_norm(inp, reuse=True, scope=scope)
        return normalized

    def __call__(self, inputs, state):
        """Long-short term memory (LSTM)."""
        with tf.variable_scope(type(self).__name__):  # "ConvLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(axis=3, num_or_size_splits=2, value=state)

            concat = _conv_linear(
                [inputs, h],
                self.filter_size,
                self._out_depth * 4,
                True,
                self._bias_initializer,
                self._kernel_initializer,
                bias_regularizer=self._weight_decay,
                kernel_regularizer=self._weight_decay,
            )

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(axis=3, num_or_size_splits=4, value=concat)

            if self._layer_norm:
                # print("using layer norm")
                i = self._norm(i, "input")
                j = self._norm(j, "transform")
                f = self._norm(f, "forget")
                o = self._norm(o, "output")

            if self._use_peepholes:
                with tf.variable_scope(
                    "peepholes", initializer=self._kernel_initializer
                ):
                    w_f_diag = tf.get_variable(
                        "w_f_diag",
                        [self.shape[0], self.shape[1], self._out_depth],
                        dtype=c.dtype,
                    )

                    w_i_diag = tf.get_variable(
                        "w_i_diag",
                        [self.shape[0], self.shape[1], self._out_depth],
                        dtype=c.dtype,
                    )

                    w_o_diag = tf.get_variable(
                        "w_o_diag",
                        [self.shape[0], self.shape[1], self._out_depth],
                        dtype=c.dtype,
                    )

            if self._use_peepholes:
                new_c = c * tf.nn.sigmoid(
                    f + self._forget_bias + w_f_diag * c
                ) + tf.nn.sigmoid(i + w_i_diag * c) * self._activation(j)
            else:
                new_c = c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(
                    i
                ) * self._activation(j)

            if self._layer_norm:
                new_c = self._norm(new_c, "state")

            if self._use_peepholes:
                new_h = self._activation(new_c) * tf.nn.sigmoid(o + w_o_diag * c)
            else:
                new_h = self._activation(new_c) * tf.nn.sigmoid(o)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat(axis=3, values=[new_c, new_h])
            return new_h, new_state


class ConvUGRNNCell(ConvRNNCell):
    """Conv UGRNN recurrent network cell."""

    def __init__(
        self,
        shape,
        filter_size,
        out_depth,
        weight_decay=0.0,
        forget_bias=1.0,
        kernel_initializer=None,
        kernel_initializer_kwargs=None,
        bias_initializer=None,
        bias_initializer_kwargs=None,
        layer_norm=False,
        norm_gain=1.0,
        norm_shift=0.0,
    ):
        """Initialize the Conv UGRNN cell.
        Args:
          shape: int tuple thats the height and width of the cell
          filter_size: int tuple thats the height and width of the filter
          out_depth: int thats the depth of the cell
          forget_bias: float, The bias added to forget gates (see above).
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
        """
        self.shape = shape
        self.filter_size = ksize(filter_size)
        self._out_depth = out_depth
        self._size = tf.TensorShape([self.shape[0], self.shape[1], self._out_depth])
        self._kernel_initializer = kernel_initializer
        if kernel_initializer_kwargs is None:
            kernel_initializer_kwargs = {}
        if kernel_initializer is not None:
            self._kernel_initializer = model_tool.initializer(
                kind=kernel_initializer, **kernel_initializer_kwargs
            )

        self._bias_initializer = bias_initializer
        if bias_initializer_kwargs is None:
            bias_initializer_kwargs = {}
        if bias_initializer is not None:
            self._bias_initializer = model_tool.initializer(
                kind=bias_initializer, **bias_initializer_kwargs
            )
        self._layer_norm = layer_norm
        self._forget_bias = forget_bias
        self._g = norm_gain
        self._b = norm_shift
        self._weight_decay = weight_decay

    @property
    def state_size(self):
        return self._size

    @property
    def output_size(self):
        return self._size

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).
        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.
        Returns:
          tensor of shape '[batch_size x shape[0] x shape[1] x out_depth]
          filled with zeros
        """
        shape = self.shape
        out_depth = self._out_depth
        zeros = tf.zeros([batch_size, shape[0], shape[1], out_depth], dtype=dtype)
        return zeros

    def _norm(self, inp, scope):
        shape = inp.get_shape()[-1:]
        gamma_init = tf.constant_initializer(self._g)
        beta_init = tf.constant_initializer(self._b)
        with tf.variable_scope(scope):
            gamma = tf.get_variable(shape=shape, initializer=gamma_init, name="gamma")
            beta = tf.get_variable(shape=shape, initializer=beta_init, name="beta")

        normalized = tf.contrib.layers.layer_norm(inp, reuse=True, scope=scope)
        return normalized

    def __call__(self, inputs, state):
        """UGRNN cell."""
        with tf.variable_scope(type(self).__name__):  # "ConvUGRNNCell"
            # Parameters of gates are concatenated into one multiply for efficiency
            concat = _conv_linear(
                [inputs, state],
                self.filter_size,
                2 * self._out_depth,
                True,
                self._bias_initializer,
                self._kernel_initializer,
                bias_regularizer=self._weight_decay,
                kernel_regularizer=self._weight_decay,
            )

            g_act, c_act = tf.split(axis=3, num_or_size_splits=2, value=concat)

            if self._layer_norm:
                g_act = self._norm(g_act, "g_act")
                c_act = self._norm(h_act, "c_act")

            c = tf.nn.tanh(c_act)
            g = tf.nn.sigmoid(g_act + self._forget_bias)
            new_state = g * state + (1.0 - g) * c
            new_output = new_state

            return new_output, new_state


class ConvIntersectionRNNCell(ConvRNNCell):
    """Conv IntersectionRNN recurrent network cell."""

    def __init__(
        self,
        shape,
        filter_size,
        out_depth,
        weight_decay=0.0,
        forget_bias=1.0,
        kernel_initializer=None,
        kernel_initializer_kwargs=None,
        bias_initializer=None,
        bias_initializer_kwargs=None,
        layer_norm=False,
        norm_gain=1.0,
        norm_shift=0.0,
    ):
        """Initialize the Conv IntersectionRNN cell.
        Args:
          shape: int tuple thats the height and width of the cell
          filter_size: int tuple thats the height and width of the filter
          out_depth: int thats the depth of the cell
          forget_bias: float, The bias added to forget gates (see above).
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
        """
        self.shape = shape
        self.filter_size = ksize(filter_size)
        self._out_depth = out_depth
        self._size = tf.TensorShape([self.shape[0], self.shape[1], self._out_depth])
        self._kernel_initializer = kernel_initializer
        if kernel_initializer_kwargs is None:
            kernel_initializer_kwargs = {}
        if kernel_initializer is not None:
            self._kernel_initializer = model_tool.initializer(
                kind=kernel_initializer, **kernel_initializer_kwargs
            )

        self._bias_initializer = bias_initializer
        if bias_initializer_kwargs is None:
            bias_initializer_kwargs = {}
        if bias_initializer is not None:
            self._bias_initializer = model_tool.initializer(
                kind=bias_initializer, **bias_initializer_kwargs
            )
        self._layer_norm = layer_norm
        self._forget_bias = forget_bias
        self._g = norm_gain
        self._b = norm_shift
        self._weight_decay = weight_decay

    @property
    def state_size(self):
        return self._size

    @property
    def output_size(self):
        return self._size

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).
        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.
        Returns:
          tensor of shape '[batch_size x shape[0] x shape[1] x out_depth]
          filled with zeros
        """
        shape = self.shape
        out_depth = self._out_depth
        zeros = tf.zeros([batch_size, shape[0], shape[1], out_depth], dtype=dtype)
        return zeros

    def _norm(self, inp, scope):
        shape = inp.get_shape()[-1:]
        gamma_init = tf.constant_initializer(self._g)
        beta_init = tf.constant_initializer(self._b)
        with tf.variable_scope(scope):
            gamma = tf.get_variable(shape=shape, initializer=gamma_init, name="gamma")
            beta = tf.get_variable(shape=shape, initializer=beta_init, name="beta")

        normalized = tf.contrib.layers.layer_norm(inp, reuse=True, scope=scope)
        return normalized

    def __call__(self, inputs, state):
        """IntersectionRNN cell."""
        with tf.variable_scope(type(self).__name__):  # "ConvIntersectionRNNCell"
            # Parameters of gates are concatenated into one multiply for efficiency
            if (
                inputs.get_shape().as_list()[1] != self.shape[0]
                or inputs.get_shape().as_list()[2] != self.shape[1]
                or inputs.get_shape().as_list()[3] != self._out_depth
            ):
                raise ValueError(
                    "Input shape {} and output shape {} must match.".format(
                        inputs.get_shape().as_list(), self.shape + [self._out_depth]
                    )
                )

            n_dim = i_dim = self._out_depth
            concat = _conv_linear(
                [inputs, state],
                self.filter_size,
                2 * n_dim + 2 * i_dim,
                True,
                self._bias_initializer,
                self._kernel_initializer,
                bias_regularizer=self._weight_decay,
                kernel_regularizer=self._weight_decay,
            )

            gh_act, h_act, gy_act, y_act = tf.split(
                axis=3, num_or_size_splits=[n_dim, n_dim, i_dim, i_dim], value=concat
            )

            if self._layer_norm:
                gh_act = self._norm(gh_act, "gh_act")
                h_act = self._norm(h_act, "h_act")
                gy_act = self._norm(gy_act, "gy_act")
                y_act = self._norm(y_act, "y_act")

            h = tf.nn.tanh(h_act)
            y = tf.nn.relu(y_act)
            gh = tf.nn.sigmoid(gh_act + self._forget_bias)
            gy = tf.nn.sigmoid(gy_act + self._forget_bias)

            new_state = gh * state + (1.0 - gh) * h  # passed through time
            new_y = gy * inputs + (1.0 - gy) * y  # passed through depth

            return new_y, new_state


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
        ds_repeat=False,
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
        data_format="channels_last",
        kernel_initializer="xavier",
        kernel_initializer_kwargs=None,
        bias_initializer=tf.zeros_initializer,
        weight_decay=None,
        layer_norm=False,
        recurrent_keep_prob=1.0,
        total_training_steps=250000,
        norm_gain=1.0,
        norm_shift=0.0,
        batch_norm=False,
        batch_norm_cell_out=False,
        batch_norm_decay=0.9,
        batch_norm_epsilon=1e-5,
        bn_trainable=True,
        crossdevice_bn_kwargs={},
        gate_tau_bn_gamma_init=0.1,
        edges_init_zero=None,
        is_training=False,
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
        self.ds_repeat = ds_repeat
        self.data_format = data_format

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

        self.recurrent_keep_prob = recurrent_keep_prob
        self.total_training_steps = total_training_steps
        self._batch_norm_func = model_tool.batchnorm_corr

        self._batch_norm = batch_norm
        self._batch_norm_cell_out = batch_norm_cell_out
        self._batch_norm_decay = batch_norm_decay
        self._batch_norm_epsilon = batch_norm_epsilon
        self._bn_trainable = bn_trainable
        self._crossdevice_bn_kwargs = crossdevice_bn_kwargs
        self._is_training = is_training
        self._gate_tau_bn_gamma_init = gate_tau_bn_gamma_init

        if edges_init_zero is None:
            if (self._feedback_activation is None) or (
                self._feedback_activation == tf.identity
            ):  # using resnet strategy since fb_input is added
                self._edges_init_zero = True
            else:
                self._edges_init_zero = False
        else:
            self._edges_init_zero = edges_init_zero

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

    def _conv(
        self,
        inp,
        filter_size,
        out_depth,
        scope,
        use_bias=True,
        bias_initializer=None,
        kernel_initializer=None,
        weight_decay=None,
        data_format="channels_last",
        is_training=True,
        batch_norm=False,
        batch_norm_decay=0.9,
        batch_norm_epsilon=1e-5,
        batch_norm_init_zero=False,
        batch_norm_constant_init=None,
        time_sep=False,
        time_suffix=None,
        bn_trainable=True,
        crossdevice_bn_kwargs={},
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
        if time_sep:
            assert time_suffix is not None

        if batch_norm:
            use_bias = False  # unnecessary in this case

        dtype = inp.dtype
        shape = inp.shape.as_list()
        if data_format == "channels_last":
            h = shape[1]
            w = shape[2]
            in_depth = shape[3]
        elif data_format == "channels_first":
            h = shape[2]
            w = shape[3]
            in_depth = shape[1]

        if filter_size[0] > h:
            filter_size[0] = h
        if filter_size[1] > w:
            filter_size[1] = w

        if weight_decay is None:
            weight_decay = 0.0
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
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            )

            out = tf.nn.conv2d(inp, kernel, strides=[1, 1, 1, 1], padding="SAME")

            if batch_norm:
                out = self._batch_norm_func(
                    inputs=out,
                    is_training=is_training,
                    data_format=data_format,
                    decay=batch_norm_decay,
                    epsilon=batch_norm_epsilon,
                    constant_init=batch_norm_constant_init,
                    init_zero=batch_norm_init_zero,
                    activation=None,
                    time_suffix=time_suffix,
                    bn_trainable=bn_trainable,
                    **crossdevice_bn_kwargs
                )

            elif use_bias:
                if bias_initializer is None:
                    bias_initializer = tf.constant_initializer(0.0, dtype=dtype)
                    bias_term = tf.get_variable(
                        "bias", [out_depth], dtype=dtype, initializer=bias_initializer
                    )

                    out = out + bias_term

            return out

    def _ds_conv(
        self,
        inp,
        filter_size,
        out_depth,
        scope,
        use_bias=True,
        repeat=False,
        intermediate_activation=tf.nn.elu,
        ch_mult=1,
        bias_initializer=None,
        kernel_initializer=None,
        weight_decay=None,
        data_format="channels_last",
        is_training=True,
        batch_norm=False,
        batch_norm_decay=0.9,
        batch_norm_epsilon=1e-5,
        batch_norm_init_zero=False,
        batch_norm_constant_init=None,
        time_sep=False,
        time_suffix=None,
        bn_trainable=True,
        crossdevice_bn_kwargs={},
    ):

        if time_sep:
            assert time_suffix is not None

        if batch_norm:
            use_bias = False  # unnecessary in this case

        ksize = [f for f in filter_size]
        dtype = inp.dtype
        shape = inp.shape.as_list()
        if data_format == "channels_last":
            h = shape[1]
            w = shape[2]
            in_depth = shape[3]
        elif data_format == "channels_first":
            h = shape[2]
            w = shape[3]
            in_depth = shape[1]

        if filter_size[0] > h:
            ksize[0] = h
        if filter_size[1] > w:
            ksize[1] = w

        if out_depth is None:
            out_depth = in_depth

        if weight_decay is None:
            weight_decay = 0.0
        if kernel_initializer is None:
            kernel_initializer = tf.contrib.layers.xavier_initializer()
        if bias_initializer is None:
            bias_initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope(scope):

            depthwise_filter = tf.get_variable(
                "depthwise_weights",
                [ksize[0], ksize[1], in_depth, ch_mult],
                dtype=dtype,
                initializer=kernel_initializer,
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            )

            pointwise_filter = tf.get_variable(
                "pointwise_weights",
                [1, 1, in_depth * ch_mult, out_depth],
                dtype=dtype,
                initializer=kernel_initializer,
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            )

            if repeat:
                depthwise_filter_0 = tf.get_variable(
                    "depthwise_weights_0",
                    [ksize[0], ksize[1], in_depth, ch_mult],
                    dtype=dtype,
                    initializer=kernel_initializer,
                    regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                )
                inp = tf.nn.depthwise_conv2d(
                    inp,
                    depthwise_filter_0,
                    strides=[1, 1, 1, 1],
                    padding="SAME",
                    name="ds_conv0",
                )
                inp = intermediate_activation(inp)

            out = tf.nn.separable_conv2d(
                inp,
                depthwise_filter,
                pointwise_filter,
                strides=[1, 1, 1, 1],
                padding="SAME",
                name="ds_conv",
            )

            if batch_norm:
                out = self._batch_norm_func(
                    inputs=out,
                    is_training=is_training,
                    data_format=data_format,
                    decay=batch_norm_decay,
                    epsilon=batch_norm_epsilon,
                    constant_init=batch_norm_constant_init,
                    init_zero=batch_norm_init_zero,
                    activation=None,
                    time_suffix=time_suffix,
                    bn_trainable=bn_trainable,
                    **crossdevice_bn_kwargs
                )
            elif use_bias:
                bias = tf.get_variable(
                    "bias", [out_depth], dtype=dtype, initializer=bias_initializer
                )
                out = out + bias

            return out

    def _drop_recurrent_step(self, hidden, keep_prob, is_training=True):

        if is_training:
            batch_size = tf.shape(hidden)[0]
            noise_shape = [batch_size, 1, 1, 1]
            random_tensor = keep_prob
            random_tensor += tf.random_uniform(shape=noise_shape, dtype=tf.float32)
            binary_tensor = tf.floor(random_tensor)
            hidden = tf.div(hidden, keep_prob) * binary_tensor
        return hidden

    def _apply_recurrent_dropout(self, inp, current_step=None):

        keep_prob = self.recurrent_keep_prob
        if keep_prob < 1.0:

            # linearly decrease keep prob over the course of training
            if current_step is None:
                current_step = tf.cast(tf.train.get_or_create_global_step(), tf.float32)
            burn_in_steps = self.total_training_steps
            current_ratio = current_step / burn_in_steps
            current_ratio = tf.minimum(1.0, current_ratio)

            keep_prob = 1 - current_ratio * (1 - keep_prob)
            inp = self._drop_recurrent_step(
                inp, keep_prob, is_training=self._is_training
            )
        return inp

    def _apply_temporal_op(
        self,
        inp,
        filter_size,
        out_depth,
        scope,
        separable,
        use_bias=True,
        batch_norm_init_zero=False,
        batch_norm_constant_init=None,
        time_sep=False,
        time_suffix=None,
    ):
        """
        Wrapper for _conv and _ds_conv that applies recurrent dropout and other cell-shared kwargs
        """
        if separable:
            inp = self._ds_conv(
                inp,
                filter_size,
                out_depth,
                scope,
                use_bias=use_bias,
                repeat=self.ds_repeat,
                intermediate_activation=tf.nn.elu,
                kernel_initializer=self._kernel_initializer,
                weight_decay=self._weight_decay,
                is_training=self._is_training,
                data_format=self.data_format,
                batch_norm=self._batch_norm,
                batch_norm_decay=self._batch_norm_decay,
                batch_norm_epsilon=self._batch_norm_epsilon,
                batch_norm_init_zero=batch_norm_init_zero,
                batch_norm_constant_init=batch_norm_constant_init,
                time_sep=time_sep,
                time_suffix=time_suffix,
                bn_trainable=self._bn_trainable,
                crossdevice_bn_kwargs=self._crossdevice_bn_kwargs,
            )
        else:  # not separable, use regular conv
            inp = self._conv(
                inp,
                filter_size,
                out_depth,
                scope,
                use_bias=use_bias,
                kernel_initializer=self._kernel_initializer,
                weight_decay=self._weight_decay,
                is_training=self._is_training,
                data_format=self.data_format,
                batch_norm=self._batch_norm,
                batch_norm_decay=self._batch_norm_decay,
                batch_norm_epsilon=self._batch_norm_epsilon,
                batch_norm_init_zero=batch_norm_init_zero,
                batch_norm_constant_init=batch_norm_constant_init,
                time_sep=time_sep,
                time_suffix=time_suffix,
                bn_trainable=self._bn_trainable,
                crossdevice_bn_kwargs=self._crossdevice_bn_kwargs,
            )

        # apply recurrent dropout
        inp = self._apply_recurrent_dropout(inp)
        return inp

    def __call__(
        self,
        inputs,
        state,
        fb_input,
        res_input,
        time_sep=False,
        time_suffix=None,
        **training_kwargs
    ):
        """
        Produce outputs of RecipCell, given inputs and previous state {'cell':cell_state, 'out':out_state}

        inputs: dict w keys ('ff', 'fb'). ff and fb inputs must have the same shape.
        """
        self._is_training = training_kwargs.get("is_training", self._is_training)
        if time_sep:
            assert time_suffix is not None

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

                self.in_depth = inputs.shape.as_list()[-1]

                if self.feedback_entry == "input" and fb_input is not None:
                    fb_input = self._apply_temporal_op(
                        fb_input,
                        self.feedback_filter_size,
                        self.in_depth,
                        separable=self.feedback_depth_separable,
                        scope="feedback",
                        batch_norm_init_zero=self._edges_init_zero,
                        time_sep=time_sep,
                        time_suffix=time_suffix,
                    )
                    inputs += self._feedback_activation(fb_input)

                inputs = self._input_activation(inputs, name="inputs")

            if self.use_cell:
                with tf.variable_scope("cell"):

                    # if cell depth and out depth are different, need to change channel number of input
                    cell_input = tf.zeros_like(
                        prev_cell, dtype=tf.float32, name="cell_input"
                    )
                    assert self.cell_residual or self.input_to_cell
                    if self.cell_residual:
                        assert res_input is not None
                    if res_input is not None and self.cell_residual:
                        cell_input += self._apply_temporal_op(
                            res_input,
                            self.ff_filter_size,
                            self.cell_depth,
                            separable=self.ff_depth_separable,
                            scope="res_to_cell",
                            time_sep=time_sep,
                            time_suffix=time_suffix,
                        )

                    if self.input_to_cell:
                        cell_input += self._apply_temporal_op(
                            inputs,
                            self.ff_filter_size,
                            self.cell_depth,
                            separable=self.ff_depth_separable,
                            scope="input_to_cell",
                            time_sep=time_sep,
                            time_suffix=time_suffix,
                        )

                    if fb_input is not None and self.feedback_entry == "cell":
                        fb_input += self._apply_temporal_op(
                            fb_input,
                            self.feedback_filter_size,
                            self.cell_depth,
                            separable=self.feedback_depth_separable,
                            scope="feedback",
                            batch_norm_init_zero=self._edges_init_zero,
                            time_sep=time_sep,
                            time_suffix=time_suffix,
                        )
                        cell_input += self._feedback_activation(fb_input)

                    # ops
                    # cell tau
                    cell_tau = self._apply_temporal_op(
                        prev_cell,
                        self.cell_tau_filter_size,
                        self.cell_depth,
                        separable=self.tau_depth_separable,
                        scope="tau",
                        batch_norm_constant_init=self._gate_tau_bn_gamma_init,
                        time_sep=time_sep,
                        time_suffix=time_suffix,
                    )

                    # cell gate
                    cell_gate = self._apply_temporal_op(
                        prev_out,
                        self.gate_filter_size,
                        self.cell_depth,
                        separable=self.gate_depth_separable,
                        scope="gate",
                        batch_norm_constant_init=self._gate_tau_bn_gamma_init,
                        time_sep=time_sep,
                        time_suffix=time_suffix,
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
                            next_cell, scope="cell_layer_norm", dtype=dtype
                        )
                    elif self._batch_norm_cell_out:
                        next_cell = self._batch_norm_func(
                            inputs=next_cell,
                            is_training=self._is_training,
                            data_format=self.data_format,
                            decay=self._batch_norm_decay,
                            epsilon=self._batch_norm_epsilon,
                            init_zero=False,
                            constant_init=None,
                            activation=None,
                            time_suffix=time_suffix,
                            bn_trainable=self._bn_trainable,
                            **self._crossdevice_bn_kwargs
                        )

                    next_cell = self._cell_activation(next_cell)

            with tf.variable_scope("out"):

                if self.input_to_out:
                    # never apply dropout here
                    if self.in_out_depth_separable:
                        out_input = self._ds_conv(
                            inputs,
                            self.in_out_filter_size,
                            out_depth=self.out_depth,
                            use_bias=True,
                            scope="input_to_out",
                            kernel_initializer=self._kernel_initializer,
                            bias_initializer=self._bias_initializer,
                            weight_decay=self._weight_decay,
                            repeat=self.ds_repeat,
                            is_training=self._is_training,
                            batch_norm=self._batch_norm,
                            batch_norm_decay=self._batch_norm_decay,
                            batch_norm_epsilon=self._batch_norm_epsilon,
                            time_sep=time_sep,
                            time_suffix=time_suffix,
                            bn_trainable=self._bn_trainable,
                            crossdevice_bn_kwargs=self._crossdevice_bn_kwargs,
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
                            dtype=dtype,
                            initializer=self._bias_initializer,
                        )

                        out_input = tf.nn.conv2d(
                            inputs,
                            in_to_out_kernel,
                            strides=[1, 1, 1, 1],
                            padding="SAME",
                            name="out_input",
                        )

                        if self._batch_norm:
                            out_input = self._batch_norm_func(
                                inputs=out_input,
                                is_training=self._is_training,
                                data_format=self.data_format,
                                decay=self._batch_norm_decay,
                                epsilon=self._batch_norm_epsilon,
                                init_zero=False,
                                constant_init=None,
                                activation=None,
                                time_suffix=time_suffix,
                                bn_trainable=self._bn_trainable,
                                **self._crossdevice_bn_kwargs
                            )
                else:
                    out_input = tf.identity(inputs, name="out_input")

                if self.cell_to_out and self.use_cell:
                    out_input += self._apply_temporal_op(
                        prev_cell,
                        self.gate_filter_size,
                        self.out_depth,
                        separable=self.gate_depth_separable,
                        scope="gate",
                        time_sep=time_sep,
                        time_suffix=time_suffix,
                    )

                if res_input is not None and self.out_residual:
                    with tf.variable_scope("res_add"):
                        out_input = residual_add(
                            out_input,
                            res_input,
                            batch_norm=self._batch_norm,
                            batch_norm_decay=self._batch_norm_decay,
                            batch_norm_epsilon=self._batch_norm_epsilon,
                            is_training=self._is_training,
                            init_zero=False,
                            sp_resize=True,
                            time_sep=time_sep,
                            time_suffix=time_suffix,
                            bn_trainable=self._bn_trainable,
                            crossdevice_bn_kwargs=self._crossdevice_bn_kwargs,
                        )

                if fb_input is not None and self.feedback_entry == "out":
                    if fb_input.shape.as_list()[1:3] == prev_out.shape.as_list()[1:3]:
                        fb_input = self._apply_temporal_op(
                            fb_input,
                            self.feedback_filter_size,
                            self.out_depth,
                            separable=self.feedback_depth_separable,
                            scope="feedback",
                            batch_norm_init_zero=self._edges_init_zero,
                            time_sep=time_sep,
                            time_suffix=time_suffix,
                        )
                        out_input += self._feedback_activation(fb_input)
                    else:
                        assert (
                            self.feedback_filter_size == fb_input.shape.as_list()[1:3]
                        )
                        fb_input = self._apply_temporal_op(
                            fb_input,
                            self.feedback_filter_size,
                            self.out_depth,
                            separable=self.feedback_depth_separable,
                            scope="feedback",
                            batch_norm_init_zero=self._edges_init_zero,
                            time_sep=time_sep,
                            time_suffix=time_suffix,
                        )
                        out_input += self._feedback_activation(fb_input)

                # ops
                out_tau = self._apply_temporal_op(
                    prev_out,
                    self.tau_filter_size,
                    self.out_depth,
                    separable=self.tau_depth_separable,
                    scope="tau",
                    batch_norm_constant_init=self._gate_tau_bn_gamma_init,
                    time_sep=time_sep,
                    time_suffix=time_suffix,
                )

                if self.use_cell and not self.cell_to_out:
                    out_gate = self._apply_temporal_op(
                        prev_cell,
                        self.gate_filter_size,
                        self.out_depth,
                        separable=self.gate_depth_separable,
                        scope="gate",
                        batch_norm_constant_init=self._gate_tau_bn_gamma_init,
                        time_sep=time_sep,
                        time_suffix=time_suffix,
                    )
                else:
                    out_gate = tf.zeros(
                        shape=out_input.shape.as_list(),
                        dtype=tf.float32,
                        name="out_gate",
                    )

                if res_input is not None and self.residual_to_out_gate:
                    out_gate += self._apply_temporal_op(
                        res_input,
                        self.gate_filter_size,
                        self.out_depth,
                        separable=self.gate_depth_separable,
                        scope="residual_to_out_gate",
                        batch_norm_constant_init=self._gate_tau_bn_gamma_init,
                        time_sep=time_sep,
                        time_suffix=time_suffix,
                    )

                out_tau = self._tau_nonlinearity(out_tau)
                out_gate = self._gate_nonlinearity(out_gate)

                next_out = (self._tau_offset + self._tau_k * out_tau) * prev_out + (
                    self._gate_offset + self._gate_k * out_gate
                ) * out_input

                if self._layer_norm:
                    next_out = self._norm(next_out, scope="out_layer_norm", dtype=dtype)
                elif self._batch_norm_cell_out:
                    next_out = self._batch_norm_func(
                        inputs=next_out,
                        is_training=self._is_training,
                        data_format=self.data_format,
                        decay=self._batch_norm_decay,
                        epsilon=self._batch_norm_epsilon,
                        init_zero=False,
                        constant_init=None,
                        activation=None,
                        time_suffix=time_suffix,
                        bn_trainable=self._bn_trainable,
                        **self._crossdevice_bn_kwargs
                    )

                next_out = self._out_activation(next_out)

                if self.use_cell:
                    next_state = tf.concat(axis=3, values=[next_cell, next_out])
                else:
                    next_state = next_out

            # return next state
            return next_out, next_state


class tnn_ConvUGRNNCell(ConvRNNCell):
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

        self.conv_cell = ConvUGRNNCell(**self.memory[1])

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
            output = self.harbor[0](
                inputs,
                self.harbor_shape,
                self.name_tmp,
                reuse=self._reuse,
                **self.harbor[1]
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
                                output, inputs, **kwargs
                            )  # component_conv needs to know the inputs
                        else:
                            output = function(
                                output, inputs, **kwargs
                            )  # component_conv needs to know the inputs
                    else:
                        output = function(output, **kwargs)
                pre_name_counter += 1

            if state is None:
                bs = output.get_shape().as_list()[0]
                state = self.conv_cell.zero_state(bs, dtype=self.dtype_tmp)

            output, state = self.conv_cell(output, state)
            self.state = tf.identity(state, name="state")

            post_name_counter = 0
            for function, kwargs in self.post_memory:
                with tf.variable_scope(
                    "post_" + str(post_name_counter), reuse=self._reuse
                ):
                    if function.__name__ == "component_conv":
                        output = function(output, inputs, **kwargs)
                    elif function.__name__ == "residual_add":
                        output = function(output, res_input, **kwargs)
                    else:
                        output = function(output, **kwargs)
                post_name_counter += 1
            self.output_tmp = tf.identity(
                tf.cast(output, self.dtype_tmp), name="output"
            )

            self._reuse = True

        self.state_shape = self.state.shape
        self.output_tmp_shape = self.output_tmp.shape
        return self.output_tmp, state

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


class tnn_ConvIntersectionRNNCell(ConvRNNCell):
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

        self.conv_cell = ConvIntersectionRNNCell(**self.memory[1])

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
            output = self.harbor[0](
                inputs,
                self.harbor_shape,
                self.name_tmp,
                reuse=self._reuse,
                **self.harbor[1]
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
                                output, inputs, **kwargs
                            )  # component_conv needs to know the inputs
                        else:
                            output = function(
                                output, inputs, **kwargs
                            )  # component_conv needs to know the inputs
                    else:
                        output = function(output, **kwargs)
                pre_name_counter += 1

            if state is None:
                bs = output.get_shape().as_list()[0]
                state = self.conv_cell.zero_state(bs, dtype=self.dtype_tmp)

            output, state = self.conv_cell(output, state)
            self.state = tf.identity(state, name="state")

            post_name_counter = 0
            for function, kwargs in self.post_memory:
                with tf.variable_scope(
                    "post_" + str(post_name_counter), reuse=self._reuse
                ):
                    if function.__name__ == "component_conv":
                        output = function(output, inputs, **kwargs)
                    elif function.__name__ == "residual_add":
                        output = function(output, res_input, **kwargs)
                    else:
                        output = function(output, **kwargs)
                post_name_counter += 1
            self.output_tmp = tf.identity(
                tf.cast(output, self.dtype_tmp), name="output"
            )

            self._reuse = True

        self.state_shape = self.state.shape
        self.output_tmp_shape = self.output_tmp.shape
        return self.output_tmp, state

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


class tnn_ConvNormBasicCell(ConvRNNCell):
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

        self.conv_cell = ConvNormBasicCell(
            memory[1]["shape"],
            memory[1]["filter_size"],
            memory[1]["out_depth"],
            memory[1]["layer_norm"],
            memory[1]["kernel_regularizer"],
            memory[1]["bias_regularizer"],
        )

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
            output = self.harbor[0](
                inputs,
                self.harbor_shape,
                self.name_tmp,
                reuse=self._reuse,
                **self.harbor[1]
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
                                output, inputs, **kwargs
                            )  # component_conv needs to know the inputs
                        else:
                            output = function(
                                output, inputs, **kwargs
                            )  # component_conv needs to know the inputs
                    else:
                        output = function(output, **kwargs)
                pre_name_counter += 1

            if state is None:
                bs = output.get_shape().as_list()[0]
                state = self.conv_cell.zero_state(bs, dtype=self.dtype_tmp)

            output, state = self.conv_cell(output, state)
            self.state = tf.identity(state, name="state")

            post_name_counter = 0
            for function, kwargs in self.post_memory:
                with tf.variable_scope(
                    "post_" + str(post_name_counter), reuse=self._reuse
                ):
                    if function.__name__ == "component_conv":
                        output = function(output, inputs, **kwargs)
                    elif function.__name__ == "residual_add":
                        output = function(output, res_input, **kwargs)
                    else:
                        output = function(output, **kwargs)
                post_name_counter += 1
            self.output_tmp = tf.identity(
                tf.cast(output, self.dtype_tmp), name="output"
            )

            self._reuse = True

        self.state_shape = self.state.shape
        self.output_tmp_shape = self.output_tmp.shape
        return self.output_tmp, state

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


class tnn_ConvGRUCell(ConvRNNCell):
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

        self.conv_cell = ConvGRUCell(**self.memory[1])

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
            output = self.harbor[0](
                inputs,
                self.harbor_shape,
                self.name_tmp,
                reuse=self._reuse,
                **self.harbor[1]
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
                                output, inputs, **kwargs
                            )  # component_conv needs to know the inputs
                        else:
                            output = function(
                                output, inputs, **kwargs
                            )  # component_conv needs to know the inputs
                    else:
                        output = function(output, **kwargs)
                pre_name_counter += 1

            if state is None:
                bs = output.get_shape().as_list()[0]
                state = self.conv_cell.zero_state(bs, dtype=self.dtype_tmp)

            output, state = self.conv_cell(output, state)
            self.state = tf.identity(state, name="state")

            post_name_counter = 0
            for function, kwargs in self.post_memory:
                with tf.variable_scope(
                    "post_" + str(post_name_counter), reuse=self._reuse
                ):
                    if function.__name__ == "component_conv":
                        output = function(output, inputs, **kwargs)
                    elif function.__name__ == "residual_add":
                        output = function(output, res_input, **kwargs)
                    else:
                        output = function(output, **kwargs)
                post_name_counter += 1
            self.output_tmp = tf.identity(
                tf.cast(output, self.dtype_tmp), name="output"
            )

            self._reuse = True

        self.state_shape = self.state.shape
        self.output_tmp_shape = self.output_tmp.shape
        return self.output_tmp, state

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


class tnn_ConvLSTMCell(ConvRNNCell):
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

        self.conv_cell = ConvLSTMCell(**self.memory[1])

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
            output = self.harbor[0](
                inputs,
                self.harbor_shape,
                self.name_tmp,
                reuse=self._reuse,
                **self.harbor[1]
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
                                output, inputs, **kwargs
                            )  # component_conv needs to know the inputs
                        else:
                            output = function(
                                output, inputs, **kwargs
                            )  # component_conv needs to know the inputs
                    else:
                        output = function(output, **kwargs)
                pre_name_counter += 1

            if state is None:
                bs = output.get_shape().as_list()[0]
                state = self.conv_cell.zero_state(bs, dtype=self.dtype_tmp)

            output, state = self.conv_cell(output, state)
            self.state = tf.identity(state, name="state")

            post_name_counter = 0
            for function, kwargs in self.post_memory:
                with tf.variable_scope(
                    "post_" + str(post_name_counter), reuse=self._reuse
                ):
                    if function.__name__ == "component_conv":
                        output = function(output, inputs, **kwargs)
                    elif function.__name__ == "residual_add":
                        output = function(output, res_input, **kwargs)
                    else:
                        output = function(output, **kwargs)
                post_name_counter += 1
            self.output_tmp = tf.identity(
                tf.cast(output, self.dtype_tmp), name="output"
            )

            self._reuse = True

        self.state_shape = self.state.shape
        self.output_tmp_shape = self.output_tmp.shape
        return self.output_tmp, state

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


class tnn_ReciprocalGateCell(ConvRNNCell):
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

        self.internal_time = 0
        self.max_internal_time = self.memory[1].get("max_internal_time", None)

        # signature: ReciprocalGateCell(shape, ff_filter_size, cell_filter_size, cell_depth, out_depth, **kwargs)
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

        mem_kwargs = copy.deepcopy(self.memory[1])
        mem_kwargs.pop("time_sep", None)
        mem_kwargs.pop("max_internal_time", None)
        self.conv_cell = ReciprocalGateCell(**mem_kwargs)

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
            curr_time_suffix = "t" + str(self.internal_time)
            pre_name_counter = 0
            for function, kwargs in self.pre_memory:
                with tf.variable_scope(
                    "pre_" + str(pre_name_counter), reuse=self._reuse
                ):
                    if kwargs.get("time_sep", False):
                        kwargs[
                            "time_suffix"
                        ] = curr_time_suffix  # used for scoping in the op

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

            if self.memory[1].get("time_sep", False):
                output, state = self.conv_cell(
                    output,
                    state,
                    fb_input,
                    res_input,
                    time_sep=True,
                    time_suffix=curr_time_suffix,
                )
            else:
                output, state = self.conv_cell(
                    output, state, fb_input, res_input, time_sep=False, time_suffix=None
                )

            self.state = tf.identity(state, name="state")

            post_name_counter = 0
            for function, kwargs in self.post_memory:
                with tf.variable_scope(
                    "post_" + str(post_name_counter), reuse=self._reuse
                ):
                    if kwargs.get("time_sep", False):
                        kwargs[
                            "time_suffix"
                        ] = curr_time_suffix  # used for scoping in the op

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

        self.state_shape = self.conv_cell.state_size()
        self.output_tmp_shape = self.output_tmp.shape

        if (self.max_internal_time is None) or (
            (self.max_internal_time is not None)
            and (self.internal_time < self.max_internal_time)
        ):
            self.internal_time += 1

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
