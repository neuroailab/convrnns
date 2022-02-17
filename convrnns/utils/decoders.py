import tensorflow as tf


def linear_normalize(tensor, axis):
    return tf.div(tensor, tf.reduce_sum(tensor, axis=axis, keepdims=True))


def softmax(tensor, axis, beta=1, trainable=False, scope="normalize"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if trainable:
            initializer = tf.initializers.random_uniform
        else:
            initializer = tf.constant_initializer(value=beta)
        beta_t = tf.get_variable(
            initializer=initializer,
            shape=1,
            dtype=tf.float32,
            name="softmax_beta",
            trainable=trainable,
        )
    # Directly from the tf.nn.softmax docs
    # Adding the trick to avoid the overflow on the softmax
    t_max = tf.reduce_max(tensor, axis=axis, keepdims=True)
    return tf.exp(beta_t * (tensor - t_max)) / tf.reduce_sum(
        tf.exp(beta_t * (tensor - t_max)), axis=axis, keepdims=True
    )


def normalize(tensor, axis, name="softmax", trainable=False, scope="normalize"):
    if name == "linear":
        return linear_normalize(tensor, axis)
    elif name == "softmax":
        return softmax(tensor, axis, trainable=trainable, scope=scope)
    else:
        raise "Normalization not implemented"


def compute_alpha(logits, trainable=False):
    # Normalize across categories
    probs = normalize(
        logits, axis=2, name="softmax", scope="probs", trainable=trainable
    )
    # Reduce along logits dimenstion, keep dims for later tensor multiplication
    # [B,T]
    max_probs = tf.reduce_max(probs, axis=2, keepdims=True, name="max_probs")
    # Normalize along the time dimension, to get the weighted average
    # [B,T]
    alpha = normalize(
        max_probs, axis=1, name="softmax", scope="alpha", trainable=trainable
    )
    alpha_ = tf.identity(alpha, name="alpha")
    return alpha


# simple
def simple_decoder(logits, normalization=None):
    # Just reads at the last timestep
    logits = tf.identity(logits, name="time_logits")
    alpha = compute_alpha(logits)
    # Normalize across categories
    if normalization is None:
        print("NOT using normalization")
        return logits[:, -1, :]
    else:
        print("Using normalization of ", normalization)
        return normalize(logits[:, -1, :], axis=1, name=normalization)


# w_avg
def weighted_average_decoder(logits, beta=1, trainable=False):
    logits = tf.identity(logits, name="time_logits")
    alpha_ = compute_alpha(logits)
    # Get the weights for the weighted average along the time dimension
    # [B,T,1000]
    alpha = softmax(logits, axis=1, beta=beta, trainable=trainable)
    alpha = tf.identity(alpha, name="w_avg_alpha")
    # Collapse the time dimension
    return tf.reduce_sum(tf.multiply(logits, alpha), axis=1)


# w_avg_t
def weighted_average_trainable_decoder(logits, trainable=False):
    logits = tf.identity(logits, name="time_logits")
    alpha_ = compute_alpha(logits)  # Only for comparison purposes
    alpha = tf.get_variable(
        initializer=tf.initializers.random_uniform,
        shape=logits.shape[1:],
        dtype=tf.float32,
        name="decoder_alpha",
        trainable=trainable,
    )
    alpha_tile = tf.tile(
        tf.expand_dims(alpha, axis=0), multiples=[logits.shape[0], 1, 1]
    )
    # Make all alphas positive, in a differentiable way
    pos_alpha = tf.nn.sigmoid(alpha_tile)
    # Normalize the alphas along the time dimension, to have them sum to 1
    norm_alpha = normalize(pos_alpha, axis=1, name="linear")
    # Since the whole thing is too large to store, but it is tiled, just look
    # at one of them
    norm_alpha = tf.identity(norm_alpha[0, :, :], name="w_avg_t_alpha")
    # Collapse the time dimension
    return tf.reduce_sum(tf.multiply(logits, norm_alpha), axis=1)


# w_time_avg
def weighted_time_average_decoder(logits, trainable=False):
    logits = tf.identity(logits, name="time_logits")
    alpha = compute_alpha(logits, trainable)
    # Keeping thisi tensor's name for compatibility
    alpha_ = tf.identity(alpha, name="w_time_avg_alpha")
    return tf.reduce_sum(tf.multiply(logits, alpha), axis=1)


# max_conf
def max_confidence_decoder(logits, trainable=False, batch_size=None):
    logits = tf.identity(logits, name="time_logits")
    alpha = compute_alpha(logits)
    # Normalize across categories
    probs = normalize(logits, axis=2, name="softmax", trainable=trainable)
    # For the most confident category, at which time does confidence peak?
    indices = tf.argmax(tf.reduce_max(probs, axis=2), axis=1, name="max_conf_time")
    if batch_size is None:
        batch_size = indices.shape[0]
    else:
        batch_size = tf.cast(batch_size, dtype=indices.dtype)  # placeholder otherwise
    idxs = tf.stack([tf.range(0, batch_size, dtype=indices.dtype), indices], axis=1)

    return tf.gather_nd(logits, idxs)


# time_max
def time_max_decoder(logits, trainable=False, batch_size=None):
    logits = tf.identity(logits, name="time_logits")
    alpha = compute_alpha(logits)
    # Normalize along logits dimension
    probs = normalize(logits, axis=2, name="softmax", trainable=trainable)
    # take argmax along the time dimension ->
    # at which time does each logit peaks [B,1000]
    time_idxs = tf.argmax(probs, axis=1, name="time_max_time")
    if batch_size is None:
        batch_size = logits.shape[0]
    else:
        batch_size = tf.cast(batch_size, dtype=time_idxs.dtype)  # placeholder otherwise
    batch_idxs = tf.reshape(
        tf.range(0, batch_size, dtype=time_idxs.dtype), shape=[batch_size, 1]
    ) * tf.ones_like(time_idxs)
    logit_idxs = tf.range(0, logits.shape[2], dtype=time_idxs.dtype) * tf.ones_like(
        time_idxs
    )
    idxs = tf.stack([batch_idxs, time_idxs, logit_idxs], axis=2)
    # gather_nd with those indices on the logits
    return tf.gather_nd(logits, idxs)


# thresh
def threshold_decoder(logits, threshold=0.9, trainable=False, batch_size=None):
    logits = tf.identity(logits, name="time_logits")
    alpha = compute_alpha(logits)
    if trainable:
        initializer = tf.initializers.random_uniform
    else:
        initializer = tf.constant_initializer(value=threshold)
    threshold_t = tf.get_variable(
        initializer=initializer,
        shape=1,
        dtype=tf.float32,
        name="decoder_threshold",
        trainable=trainable,
    )
    # Normalize across categories
    # Resulting shape: # [B,T,N]
    probs = normalize(logits, axis=2, name="softmax", trainable=trainable)
    # Get the max value for a given category
    # Resulting shape: # [B,T]
    max_p = tf.reduce_max(probs, axis=2)
    # Get the time index where we get the maximum probabilities, use
    # this if we do not cross the threshold
    max_idxs = tf.tile(
        tf.argmax(
            tf.reduce_max(probs, axis=2, keepdims=True), axis=1, output_type=tf.int32
        ),
        multiples=[1, probs.shape[1]],
    )
    # times at which the threshold is crossed for each example,
    # Matrix with either the time index of the value that crossed the threshold
    # or the time index of the maximum probability value
    time_idxs = tf.where(
        tf.greater_equal(max_p, threshold_t),
        x=tf.range(max_p.shape[1]) * tf.ones_like(max_p, dtype=tf.int32),
        y=max_idxs,
    )
    # Get the first time it is crossed, for every example
    indices = tf.reduce_min(time_idxs, axis=1, name="thresh_time")
    # Slice the logits appropriately
    if batch_size is None:
        batch_size = indices.shape[0]
    else:
        batch_size = tf.cast(batch_size, dtype=indices.dtype)  # placeholder otherwise

    idxs = tf.stack([tf.range(0, batch_size, dtype=indices.dtype), indices], axis=1)
    return tf.gather_nd(logits, idxs)


def mlp_decoder(logits, layers_list=[250], trainable=False, weight_decay=None):
    if weight_decay is None:
        weight_decay = 0.0

    print("Using an mlp decoder")
    print("layers_list:{}".format(layers_list))
    print("weight_decay:{}".format(weight_decay))
    print("trainable:{}".format(trainable))

    logits = tf.identity(logits, name="time_logits")
    alpha = compute_alpha(logits)
    inputs = tf.reshape(logits, shape=[-1, logits.shape[1] * logits.shape[2]])
    i = 1
    for num_neurons in layers_list:
        inputs = tf.contrib.layers.fully_connected(
            inputs,
            num_outputs=num_neurons,
            activation_fn=tf.nn.elu,
            scope="mlp_decoder/fc{}".format(i),
        )
        i += 1
    num_neurons = logits.shape[2]
    output = tf.contrib.layers.fully_connected(
        inputs,
        num_outputs=int(num_neurons),
        activation_fn=None,
        weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
        biases_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
        trainable=True,
        scope="mlp_decoder/fc_final",
    )
    return tf.reshape(output, shape=[-1, logits.shape[2]], name=None)


def conv1d_decoder(
    logits,
    activation_conv=None,
    activation_fc=None,
    pool_type="max",
    num_filters=4,
    window_lengths=range(2, 7),
    weight_decay=1e-4,
):
    logits = tf.identity(logits, name="time_logits")

    if activation_conv is not None:
        activation_conv = getattr(tf.nn, activation_conv)

    if activation_fc is not None:
        activation_fc = getattr(tf.nn, activation_fc)

    if weight_decay is None:
        weight_decay = 0.0

    if not isinstance(num_filters, list):
        num_filters = [num_filters] * len(window_lengths)

    agg_out = []
    for w_idx, w in enumerate(window_lengths):
        # (batch, time, logits) --> (batch, new_time, num_filters)
        curr_out = tf.layers.conv1d(
            logits,
            filters=num_filters[w_idx],
            kernel_size=window_lengths[w_idx],
            strides=1,
            padding="valid",
            data_format="channels_last",
            activation=activation_conv,
            use_bias=True,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            trainable=True,
            name="conv1_decoder/c_" + str(w),
        )

        # (batch, new_time, num_filters) --> (batch, num_filters)
        if pool_type == "max":
            curr_pool_out = tf.reduce_max(curr_out, axis=1, keepdims=False)
        elif pool_type == "avg":
            curr_pool_out = tf.reduce_mean(curr_out, axis=1, keepdims=False)
        else:
            raise ValueError

        agg_out.append(curr_pool_out)

    agg_out = tf.concat(agg_out, axis=-1)

    num_neurons = logits.shape[2]
    output = tf.contrib.layers.fully_connected(
        agg_out,
        num_outputs=int(num_neurons),
        activation_fn=activation_fc,
        weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
        biases_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
        trainable=True,
        scope="conv1_decoder/fc_final",
    )
    return output


def temporal_decoder(
    logits_list,
    name="simple",
    trainable=False,
    mlp_layers=[250],
    activation_conv=None,
    activation_fc=None,
    pool_type="max",
    num_filters=4,
    window_lengths=range(2, 7),
    weight_decay=1e-4,
    batch_size=None,
):
    if isinstance(logits_list, list):
        logits = tf.stack(logits_list, axis=1)
    else:
        logits = logits_list
    # Select the decoder to use
    if name == "simple_linear":
        return simple_decoder(logits, normalization="linear")
    elif name == "simple_soft":
        return simple_decoder(logits, normalization="softmax")
    elif name == "simple":
        # Last timestep with no normalization
        return simple_decoder(logits)
    elif name == "w_avg":
        # Trainable on the softmax beta
        return weighted_average_decoder(logits, trainable=trainable)
    elif name == "max_conf":
        # Trainable on the softmax beta
        return max_confidence_decoder(
            logits, trainable=trainable, batch_size=batch_size
        )
    elif name == "time_max":
        # Trainable on the softmax betas
        return time_max_decoder(logits, trainable=trainable, batch_size=batch_size)
    elif name == "thresh":
        # Trainable on the softmax beta and the theshold
        return threshold_decoder(logits, trainable=trainable, batch_size=batch_size)
    elif name == "w_avg_t":
        # Trainable on the alphas
        return weighted_average_trainable_decoder(logits, trainable=trainable)
    elif name == "w_time_avg":
        # Trainable on the two softmax betas
        return weighted_time_average_decoder(logits, trainable=trainable)
    elif name == "mlp":
        # Trainable always
        return mlp_decoder(logits, layers_list=mlp_layers, weight_decay=weight_decay)
    elif name == "conv1d":
        return conv1d_decoder(
            logits,
            activation_conv=activation_conv,
            activation_fc=activation_fc,
            pool_type=pool_type,
            num_filters=num_filters,
            window_lengths=window_lengths,
            weight_decay=weight_decay,
        )
    else:
        raise "Decoder not implemented"
