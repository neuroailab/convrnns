import tensorflow as tf
import numpy as np
import re, copy, numbers
from tensorflow.python.framework import ops
from convrnns.utils import model_tool, spatial_transformer


def _conv_linear(
    args,
    filter_size,
    out_depth,
    bias,
    bias_initializer=None,
    kernel_initializer=None,
    bias_regularizer=None,
    kernel_regularizer=None,
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
    total_arg_size_depth = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 4:
            raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
        if not shape[3]:
            raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
        else:
            total_arg_size_depth += shape[3]

    dtype = [a.dtype for a in args][0]

    if kernel_regularizer is None:
        kernel_regularizer = 0.0
    if bias_regularizer is None:
        bias_regularizer = 0.0

    # Now the computation.
    kernel = tf.get_variable(
        "weights",
        [filter_size[0], filter_size[1], total_arg_size_depth, out_depth],
        dtype=dtype,
        initializer=kernel_initializer,
        regularizer=tf.contrib.layers.l2_regularizer(kernel_regularizer),
    )
    if len(args) == 1:
        res = tf.nn.conv2d(args[0], kernel, strides=[1, 1, 1, 1], padding="SAME")
    else:
        res = tf.nn.conv2d(
            tf.concat(axis=3, values=args), kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
    if not bias:
        return res
    if bias_initializer is None:
        bias_initializer = tf.constant_initializer(0.0, dtype=dtype)
    bias_term = tf.get_variable(
        "bias",
        [out_depth],
        dtype=dtype,
        initializer=bias_initializer,
        regularizer=tf.contrib.layers.l2_regularizer(bias_regularizer),
    )
    return res + bias_term


def ksize(val):
    if isinstance(val, float):
        return [int(val), int(val)]
    elif isinstance(val, int):
        return [val, val]
    else:
        return val


def _ds_conv(
    inp,
    filter_size,
    bias,
    scope,
    out_depth=None,
    ch_mult=1,
    bias_initializer=None,
    kernel_initializer=None,
    weight_decay=None,
    data_format="channels_last",
):

    ksize = [f for f in filter_size]
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

        out = tf.nn.separable_conv2d(
            inp,
            depthwise_filter,
            pointwise_filter,
            strides=[1, 1, 1, 1],
            padding="SAME",
            data_format=data_format,
            name="ds_conv",
        )

        if not bias:
            return out
        else:

            bias = tf.get_variable(
                "bias", [out_depth], dtype=dtype, initializer=bias_initializer
            )
            return out + bias


def laplacian_regularizer(scale, scope=None):
    """Compute loss term by filtering a rank-4 tensor with the discrete Laplacian kernel.
    Takes the root-sum-of-squares across space, then sums across the out-channel dimension.

    weights: a rank 4 tensor that represents a filter for a spatial input. Shape: [H, W, D, Num_output_channels]
    scale: a real scalar to multiply the loss"""

    if isinstance(scale, numbers.Integral):
        raise ValueError("scale cannot be an integer: %s" % scale)
    if isinstance(scale, numbers.Real):
        if scale < 0.0:
            raise ValueError("Setting a scale less than 0 on a regularizer: %g" % scale)
        if scale == 0.0:
            return lambda _: None

    def laplacian_loss(weights, name=None):

        with ops.name_scope(scope, "laplacian_regularizer", [weights]) as name:
            my_scale = ops.convert_to_tensor(
                scale, dtype=weights.dtype.base_dtype, name="scale"
            )

            # for spatial/feature-factored readout weights, the mask input is shape [H, W, out_channels, 1]
            if weights.get_shape().as_list()[-1] == 1:
                weights = tf.transpose(
                    weights, perm=(0, 1, 3, 2)
                )  # put in order [H, W, 1, out_channels]

            # weights for readout have shape [h, w, d, out_channels]
            weights = tf.transpose(
                weights, perm=(3, 0, 1, 2)
            )  # out_ch treated as "batch" dimension of a convolution
            ch_in = weights.get_shape().as_list()[-1]
            L = tf.constant(
                value=[[0.5, 1, 0.5], [1, -6, 1], [0.5, 1, 0.5]], dtype=tf.float32
            )  # 2D Laplacian kernel of shape [3,3]
            L = tf.reshape(L, shape=[3, 3, 1, 1])
            L = (
                tf.zeros(shape=[3, 3, ch_in, 1], dtype=tf.float32) + L
            )  # Broadcast L to ch_in copies of the same kernel

            # Now compute loss as L2 on the output of Laplacian filtering
            conv = tf.nn.depthwise_conv2d(
                weights, L, strides=[1, 1, 1, 1], padding="VALID"
            )

            # Square and sum across H, W, D of weights; take sqrt; take sum across neurons_to_fit
            loss = tf.reduce_sum(tf.square(conv))
            loss = tf.scalar_mul(my_scale, loss)
            return tf.identity(loss, name=name)

    return laplacian_loss


def group_sparsity_regularizer(scale, scope=None):
    """Compute loss term that is L2 over H,W spatial dimensions and L1 over D and out_channels.

    weights: a rank 4 tensor that represents a filter for a spatial input. Shape: [H, W, D, Num_output_channels]
    scale: a real scalar to multiply the loss"""

    if isinstance(scale, numbers.Integral):
        raise ValueError("scale cannot be an integer: %s" % scale)
    if isinstance(scale, numbers.Real):
        if scale < 0.0:
            raise ValueError("Setting a scale less than 0 on a regularizer: %g" % scale)
        if scale == 0.0:
            return lambda _: None

    def group_sparsity_loss(weights, name=None):

        with ops.name_scope(scope, "group_sparsity_regularizer", [weights]) as name:
            my_scale = ops.convert_to_tensor(
                scale, dtype=weights.dtype.base_dtype, name="scale"
            )

            weights = tf.transpose(weights, perm=(3, 0, 1, 2))
            channelwise_l2 = tf.sqrt(
                tf.reduce_sum(tf.square(weights), axis=(1, 2))
            )  # sum across H, W spatial dimensions
            loss = tf.scalar_mul(
                my_scale, tf.reduce_sum(channelwise_l2)
            )  # sum across D and out_channels, i.e. L1 on D
            return tf.identity(loss, name=name)

    return group_sparsity_loss


def _get_regularizer(reg_scales=None):
    """
    Helper function to construct a tensorflow regularizer from a dict of scales to apply to each
    """

    if reg_scales is None:
        return None
    else:
        scale_l2 = reg_scales.get("weight_decay", 0.0)
        scale_l1 = reg_scales.get("l1", 0.0)
        scale_lap = reg_scales.get("laplacian", 0.0)
        scale_group = reg_scales.get("group_sparsity", 0.0)

        regs = []
        if scale_l2:
            regs.append(tf.contrib.layers.l2_regularizer(scale_l2))
        if scale_l1:
            regs.append(tf.contrib.layers.l1_regularizer(scale_l1))
        if scale_lap:
            regs.append(laplacian_regularizer(scale_lap))
        if scale_group:
            regs.append(group_sparsity_regularizer(scale_group))

        reg_func = tf.contrib.layers.sum_regularizer(regs)

    return reg_func


def gather_inputs(inputs, shape, l1_inpnm, ff_inpnm, node_nms):
    """Helper function that returns the skip, feedforward, and feedback inputs"""
    assert ff_inpnm is not None
    assert node_nms is not None

    if l1_inpnm not in node_nms:
        node_nms = [l1_inpnm] + node_nms  # easy to forget to add this

    # determine the skip and possible feedback inputs to this layer
    ff_idx = 0
    for idx, elem in enumerate(node_nms):
        if elem == ff_inpnm:
            ff_idx = idx
            break

    skips = node_nms[:ff_idx]  # exclude ff input
    feedbacks = node_nms[
        ff_idx + 2 :
    ]  # exclude ff input and itself, note no layer has ff_idx to be the last element

    skip_ins = []
    feedback_ins = []
    ff_in = None
    for inp in inputs:
        pat = re.compile(":|/")
        if l1_inpnm not in inp.name:
            nm = pat.sub("__", inp.name.split("/")[-2].split("_")[0])
        else:
            nm = l1_inpnm

        if ff_inpnm == nm:
            ff_in = inp
        elif nm in feedbacks:  # a feedback input
            if len(inp.shape) == 4:  # flatten conv inputs to pass through mlp later
                #                reshaped_inp = tf.reshape(inp, [inp.get_shape().as_list()[0], -1])
                feedback_ins.append(inp)
            elif len(inp.shape) == 2:
                feedback_ins.append(inp)
            else:
                raise ValueError
        elif nm in skips:
            skip_ins.append(inp)

    return ff_in, skip_ins, feedback_ins


def input_aggregator(
    inputs,
    shape,
    spatial_op,
    channel_op,
    kernel_init="xavier",
    weight_decay=None,
    reuse=None,
    ff_inpnm=None,
    ksize=3,
    activation=None,
    kernel_init_kwargs=None,
    padding="SAME",
    out_depth_per_input=None,
):
    """Helper function that combines the inputs appropriately based on the spatial and channel_ops"""

    outputs = []
    for inp in inputs:
        if len(shape) == 2:
            pat = re.compile(":|/")
            if len(inp.shape) == 2:
                if channel_op != "concat" and inp.shape[1] != shape[1]:
                    nm = pat.sub("__", inp.name.split("/")[-2].split("_")[0])
                    nm = "fc_to_fc_harbor_for_%s" % nm
                    with tf.variable_scope(nm, reuse=reuse):
                        inp = model_tool.fc(
                            inp,
                            shape[1],
                            kernel_init=kernel_init,
                            kernel_init_kwargs=kernel_init_kwargs,
                            weight_decay=weight_decay,
                            activation=activation,
                            batch_norm=False,
                        )

                outputs.append(inp)

            elif len(inp.shape) == 4:
                out = tf.reshape(
                    inp, tf.cast([inp.get_shape().as_list()[0], -1], dtype=tf.int32)
                )
                if channel_op != "concat" and out.shape[1] != shape[1]:
                    nm = pat.sub("__", inp.name.split("/")[-2].split("_")[0])
                    nm = "conv_to_fc_harbor_for_%s" % nm
                    with tf.variable_scope(nm, reuse=reuse):
                        out = model_tool.fc(
                            out,
                            shape[1],
                            kernel_init=kernel_init,
                            kernel_init_kwargs=kernel_init_kwargs,
                            weight_decay=weight_decay,
                            activation=activation,
                            batch_norm=False,
                        )

                outputs.append(out)
            else:
                raise ValueError

        elif len(shape) == 4:
            pat = re.compile(":|/")
            if len(inp.shape) == 2:
                nchannels = shape[3]
                old_channels = inp.shape[1]
                if nchannels != old_channels:
                    nm = pat.sub("__", inp.name.split("/")[-2].split("_")[0])
                    nm = "fc_to_conv_harbor_for_%s" % nm
                    with tf.variable_scope(nm, reuse=reuse):
                        inp = model_tool.fc(
                            inp,
                            nchannels,
                            kernel_init=kernel_init,
                            kernel_init_kwargs=kernel_init_kwargs,
                            weight_decay=weight_decay,
                            activation=activation,
                            batch_norm=False,
                        )

                if spatial_op == "emphasis":
                    if activation == "softmax" and nchannels != old_channels:
                        # softmax has already been applied to the fc
                        # so now we multiply by nchannels to keep mean value as 1
                        channel_normalizer = tf.cast(nchannels, dtype=tf.float32)
                        inp = tf.multiply(channel_normalizer, inp)

                # we may choose a different activation (like relu) and/or
                # we did not need to learn an fc above so we directly apply the fc
                # to the conv input, but in all cases we tile
                xs, ys = shape[1:3]
                inp = tf.tile(inp, [1, xs * ys])
                out = tf.reshape(
                    inp,
                    tf.cast(
                        (inp.shape.as_list()[0], xs, ys, nchannels), dtype=tf.int32
                    ),
                )

            elif len(inp.shape) == 4:
                if spatial_op == "tile":
                    out = tile_func(inp, shape)
                elif spatial_op == "pad":
                    out = tf.map_fn(
                        lambda im: tf.image.resize_image_with_crop_or_pad(
                            im, shape[1], shape[2]
                        ),
                        inp,
                        dtype=tf.float32,
                    )
                elif spatial_op == "sp_transform":
                    out = transform_func(
                        inp,
                        shape=shape,
                        weight_decay=weight_decay,
                        ff_inpnm=ff_inpnm,
                        reuse=reuse,
                    )
                elif spatial_op == "flatten":
                    out = tf.reshape(
                        inp, tf.cast([inp.get_shape().as_list()[0], -1], dtype=tf.int32)
                    )
                elif spatial_op == "deconv":
                    out = deconv(
                        inp,
                        shape=shape,
                        weight_decay=weight_decay,
                        ksize=ksize,
                        activation=activation,
                        padding=padding,
                        reuse=reuse,
                    )
                elif spatial_op == "factored_fc":
                    nm = pat.sub("__", inp.name.split("/")[-2].split("_")[0])
                    nm = "factored_fc_harbor_for_" + nm
                    with tf.variable_scope(nm, reuse=reuse):
                        assert out_depth_per_input is not None
                        out = factored_fc(
                            inp,
                            out_depth=out_depth_per_input,
                            spatial_mask_init=kernel_init,
                            spatial_mask_init_kwargs=kernel_init_kwargs,
                            feature_kernel_init=kernel_init,
                            feature_kernel_init_kwargs=kernel_init_kwargs,
                            activation=activation,
                            flatten=False,
                            bias=0.0,
                        )
                else:
                    out = tf.image.resize_images(inp, shape[1:3], align_corners=True)

                if (
                    channel_op != "concat"
                    and out.shape[3] != shape[3]
                    and spatial_op != "factored_fc"
                ):
                    nm = pat.sub("__", inp.name.split("/")[-2].split("_")[0])
                    nm = "conv_to_conv_harbor_for_%s" % nm
                    with tf.variable_scope(nm, reuse=reuse):
                        out = model_tool.conv(
                            out,
                            out_depth=shape[3],
                            ksize=[1, 1],
                            kernel_init=kernel_init,
                            kernel_init_kwargs=kernel_init_kwargs,
                            weight_decay=weight_decay,
                            activation=activation,
                            batch_norm=False,
                        )
            else:
                raise ValueError

            if inp.name == ff_inpnm:
                outputs.insert(0, out)
            else:
                outputs.append(out)

        else:
            raise ValueError("harbor cannot process layer of dim {}".format(len(shape)))

    if channel_op == "add":
        output = tf.add_n(outputs, name="harbor")
    elif channel_op == "multiply":
        if len(outputs) == 1:
            output = outputs[0]
        else:
            output = tf.multiply(outputs[0], outputs[1])
            if len(outputs) > 2:
                for output_elem in outputs[2:]:
                    output = tf.multiply(output, output_elem)
    else:
        output = tf.concat(outputs, axis=-1, name="harbor")

    return output


def crop_func(
    inputs, l1_inpnm, ff_inpnm, node_nms, shape, kernel_init, channel_op, reuse
):
    # note: e.g. node_nms = ['split', 'V1', 'V2', 'V4', 'pIT', 'aIT']

    ff_in, skip_ins, feedback_ins = gather_inputs(
        inputs, shape, l1_inpnm, ff_inpnm, node_nms
    )
    not_ff = feedback_ins + skip_ins
    if (
        len(not_ff) == 0 or ff_in is None or len(shape) != 4 or len(ff_in.shape) != 4
    ):  # we do nothing in this case, and proceed as usual (appeases initialization too)
        return inputs
    # combine skips and feedbacks to learn the crop from
    not_ff_ins = tf.concat(not_ff, axis=-1, name="comb")
    mlp_nm = "crop_mlp_for_%s" % ff_inpnm
    with tf.variable_scope(mlp_nm, reuse=reuse):
        mlp_out = model_tool.fc(
            not_ff_ins, 5, kernel_init=kernel_init, activation=None, batch_norm=False
        )

    alpha = tf.slice(mlp_out, [0, 0], [-1, 1])
    alpha = tf.expand_dims(tf.expand_dims(alpha, axis=-1), axis=-1)
    alpha = tf.nn.tanh(alpha)  # we want to potentially have negatives to downweight
    boxes = tf.slice(mlp_out, [0, 1], [-1, 4])
    boxes = tf.nn.sigmoid(boxes)  # keep values in [0, 1] range
    # dimensions of original ff
    total_height = tf.constant(ff_in.get_shape().as_list()[1], dtype=tf.float32)
    total_width = tf.constant(ff_in.get_shape().as_list()[2], dtype=tf.float32)
    total_depth = tf.constant(ff_in.get_shape().as_list()[3], dtype=tf.int32)
    # compute bbox coords
    offset_height_frac = tf.squeeze(tf.slice(boxes, [0, 0], [-1, 1]), axis=-1)
    offset_height = tf.floor(total_height * offset_height_frac)
    offset_width_frac = tf.squeeze(tf.slice(boxes, [0, 1], [-1, 1]), axis=-1)
    offset_width = tf.floor(total_width * offset_width_frac)
    target_height_frac = tf.squeeze(tf.slice(boxes, [0, 2], [-1, 1]))
    target_height = tf.floor(total_height * target_height_frac)
    target_width_frac = tf.squeeze(tf.slice(boxes, [0, 3], [-1, 1]))
    target_width = tf.floor(total_width * target_width_frac)
    # clip height and width of bounding box
    height_val = tf.minimum(offset_height + target_height, total_height)
    width_val = tf.minimum(offset_width + target_width, total_width)
    clipped_target_height = height_val - offset_height
    clipped_target_width = width_val - offset_width
    rem_height = total_height - height_val
    rem_width = total_width - width_val
    # construct mask
    offset_height = tf.cast(offset_height, tf.int32)
    offset_width = tf.cast(offset_width, tf.int32)
    clipped_target_height = tf.cast(clipped_target_height, tf.int32)
    clipped_target_width = tf.cast(clipped_target_width, tf.int32)
    rem_height = tf.cast(rem_height, tf.int32)
    rem_width = tf.cast(rem_width, tf.int32)
    elems = (
        offset_height,
        offset_width,
        clipped_target_height,
        clipped_target_width,
        rem_height,
        rem_width,
    )
    mask = tf.map_fn(
        lambda x: tf.pad(
            tf.ones([x[2], x[3], total_depth]),
            [[x[0], x[4]], [x[1], x[5]], [0, 0]],
            "CONSTANT",
        ),
        elems,
        dtype=tf.float32,
    )

    padded_img = tf.multiply(ff_in, mask)
    padded_img = tf.multiply(alpha, padded_img)
    pat = re.compile(":|/")
    ff_nm = pat.sub("__", ff_in.name.split("/")[-2].split("_")[0])
    new_name = ff_nm + "_mod"
    new_in = tf.add(ff_in, padded_img, name=new_name)

    new_out = [new_in]
    return new_out


def tile_func(inp, shape):
    inp_height = inp.get_shape().as_list()[1]
    inp_width = inp.get_shape().as_list()[2]
    height_multiple = 1 + (shape[1] // inp_height)
    width_multiple = 1 + (shape[2] // inp_width)
    tiled_out = tf.tile(inp, [1, height_multiple, width_multiple, 1])
    return tf.map_fn(
        lambda im: tf.image.resize_image_with_crop_or_pad(im, shape[1], shape[2]),
        tiled_out,
        dtype=tf.float32,
    )


def transform_func(inp, shape, weight_decay, ff_inpnm, reuse):
    """Learn an affine transformation on the input inp if it is a feedback or skip"""
    pat = re.compile(":|/")
    orig_nm = pat.sub("__", inp.name.split("/")[-2].split("_")[0])
    assert ff_inpnm is not None
    if ff_inpnm in orig_nm:
        return tf.image.resize_images(
            inp, shape[1:3]
        )  # simply do nothing with feedforward input
    else:
        nm = "spatial_transform_for_%s" % orig_nm
        with tf.variable_scope(nm, reuse=reuse):
            resh = tf.reshape(inp, [inp.get_shape().as_list()[0], -1], name="reshape")
            in_depth = resh.get_shape().as_list()[-1]
            if weight_decay is None:
                weight_decay = 0.0

            # identity initialization (weights = zeros and biases = identity)
            kernel = tf.get_variable(
                initializer=tf.zeros_initializer(),
                shape=[in_depth, 6],
                dtype=tf.float32,
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                name="weights",
            )

            initial_theta = np.array([[1.0, 0, 0], [0, 1.0, 0]])
            initial_theta = initial_theta.astype("float32")
            initial_theta = initial_theta.flatten()
            biases = tf.get_variable(
                initializer=tf.constant_initializer(value=initial_theta),
                shape=[6],
                dtype=tf.float32,
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                name="bias",
            )

            fcm = tf.matmul(resh, kernel)
            loc_out = tf.nn.bias_add(fcm, biases, name="loc_out")
            out_size = (shape[1], shape[2])
            assert (
                len(inp.shape) == 4
            )  # we can only spatial transform on a convolutional input
            h_trans = spatial_transformer.transformer(inp, loc_out, out_size)
            bs = inp.get_shape().as_list()[0]
            cs = inp.get_shape().as_list()[-1]
            h_trans.set_shape([bs, shape[1], shape[2], cs])
            return h_trans


def deconv(
    inp,
    shape,
    weight_decay=None,
    ksize=[3, 3],
    activation="relu",
    padding="SAME",
    reuse=None,
):
    pat = re.compile(":|/")
    if len(inp.name.split("/")) == 1:
        orig_nm = pat.sub("__", inp.name.split("/")[-1].split("_")[0])
    else:
        orig_nm = pat.sub("__", inp.name.split("/")[-2].split("_")[0])

    if (
        len(shape) == len(inp.get_shape().as_list()) - 1
    ):  # include batch dimension automatically
        shape = shape.insert(0, inp.get_shape().as_list()[0])

    if (
        inp.shape[1] == shape[1]
        and inp.shape[2] == shape[2]
        and inp.shape[3] == shape[3]
    ):
        return tf.image.resize_images(
            inp, shape[1:3], align_corners=True
        )  # simply do nothing with feedforward input or inputs of the same shape
    elif (
        inp.shape[1] > shape[1] or inp.shape[2] > shape[2]
    ):  # e.g. if connection is a skip
        nm = "deconv_for_%s" % orig_nm
        with tf.variable_scope(nm, reuse=reuse):
            if weight_decay is None:
                weight_decay = 0.0
            if isinstance(ksize, int):
                ksize = [ksize, ksize]
            in_ch = inp.get_shape().as_list()[-1]
            out_ch = shape[3]

            kernel = tf.get_variable(
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[ksize[0], ksize[1], in_ch, out_ch],
                dtype=tf.float32,
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                name="weights",
            )

            biases = tf.get_variable(
                initializer=tf.zeros_initializer(),
                shape=[out_ch],
                dtype=tf.float32,
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                name="bias",
            )

            # simply do a size-matching conv op
            stride_0 = inp.get_shape()[1] // shape[1]
            stride_1 = inp.get_shape()[2] // shape[2]
            if (
                inp.get_shape().as_list()[1] > shape[1]
                and inp.get_shape().as_list()[1] % shape[1] != 0
            ):  # e.g. 7 // 4 = 1, but should be 2, probably so we want to round up in non-divisible cases
                stride_0 += 1
            if (
                inp.get_shape().as_list()[2] > shape[2]
                and inp.get_shape().as_list()[2] % shape[2] != 0
            ):  # e.g. 7 // 4 = 1, but should be 2, probably so we want to round up in non-divisible cases
                stride_1 += 1
            conv = tf.nn.conv2d(
                inp, kernel, strides=[1, stride_0, stride_1, 1], padding=padding
            )

            output = tf.nn.bias_add(conv, biases, name="deconv_out")
            if activation is not None:
                output = getattr(tf.nn, activation)(output, name=activation)
            return output

    else:  # a feedback that requires transposed convolution
        nm = "deconv_for_%s" % orig_nm
        with tf.variable_scope(nm, reuse=reuse):
            if weight_decay is None:
                weight_decay = 0.0
            if isinstance(ksize, int):
                ksize = [ksize, ksize]
            in_ch = inp.get_shape().as_list()[-1]
            out_ch = shape[3]
            kernel = tf.get_variable(
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[ksize[0], ksize[1], out_ch, in_ch],
                dtype=tf.float32,
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                name="weights",
            )

            biases = tf.get_variable(
                initializer=tf.zeros_initializer(),
                shape=[out_ch],
                dtype=tf.float32,
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                name="bias",
            )

            stride_0 = shape[1] // inp.get_shape().as_list()[1]
            stride_1 = shape[2] // inp.get_shape().as_list()[2]
            if (
                shape[1] > inp.get_shape().as_list()[1]
                and shape[1] % inp.get_shape().as_list()[1] != 0
            ):  # e.g. 7 // 4 = 1, but should be 2, probably so we want to round up in non-divisible cases
                stride_0 += 1
            if (
                shape[2] > inp.get_shape().as_list()[2]
                and shape[2] % inp.get_shape().as_list()[2] != 0
            ):  # e.g. 7 // 4 = 1, but should be 2, probably so we want to round up in non-divisible cases
                stride_1 += 1
            conv_t = tf.nn.conv2d_transpose(
                inp,
                kernel,
                output_shape=shape,
                strides=[1, stride_0, stride_1, 1],
                padding=padding,
            )

            output = tf.nn.bias_add(conv_t, biases, name="deconv_out")
            if activation is not None:
                output = getattr(tf.nn, activation)(output, name=activation)
            return output


def sptransform_preproc(
    inputs,
    l1_inpnm,
    ff_inpnm,
    node_nms,
    shape,
    spatial_op,
    channel_op,
    kernel_init,
    weight_decay,
    dropout,
    reuse,
):
    """Learn an affine transformation on the feedforward inputs (including skips) using the feedbacks
    into that layer"""
    ff_in, skip_ins, feedback_ins = gather_inputs(
        inputs, shape, l1_inpnm, ff_inpnm, node_nms
    )
    new_inputs = [ff_in]
    not_ff = feedback_ins + skip_ins
    # Note you must pass to_exclude to the init_nodes method in main.py if using the concat channel op
    # since the total number of channels excludes the feedback channels as we are not combining them
    # otherwise it does not need to change since the number of input channels is unchanged
    if channel_op == "concat":
        print("Make sure to exclude feedback nodes in the main.init_nodes() method!")

    if (
        len(not_ff) == 0 or ff_in is None or len(shape) != 4 or len(ff_in.shape) != 4
    ):  # we do nothing in this case, and proceed as usual (appeases initialization too)
        out_val = input_aggregator(
            inputs,
            shape,
            spatial_op,
            channel_op,
            kernel_init,
            weight_decay,
            reuse,
            ff_inpnm,
        )
        return out_val

    # aggregate feedforward input
    ff_out = input_aggregator(
        new_inputs,
        shape,
        spatial_op,
        channel_op,
        kernel_init,
        weight_decay,
        reuse,
        ff_inpnm,
    )
    # combine skips and feedbacks to learn the affine transform from
    not_ff_ins = tf.concat(not_ff, axis=-1, name="comb")
    if dropout is not None:
        not_ff_ins = tf.nn.dropout(not_ff_ins, keep_prob=dropout)
    mlp_nm = "spatial_transform_for_%s" % ff_inpnm
    # we use the feedbacks to learn our localizer network
    with tf.variable_scope(mlp_nm, reuse=reuse):
        in_depth = not_ff_ins.get_shape().as_list()[-1]

        # identity initialization (weights = zeros and biases = identity)
        kernel = tf.get_variable(
            initializer=tf.zeros_initializer(),
            shape=[in_depth, 6],
            dtype=tf.float32,
            name="weights",
        )

        initial_theta = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        initial_theta = initial_theta.astype("float32")
        initial_theta = initial_theta.flatten()
        biases = tf.get_variable(
            initializer=tf.constant_initializer(value=initial_theta),
            shape=[6],
            dtype=tf.float32,
            name="bias",
        )

        fcm = tf.matmul(not_ff_ins, kernel)
        loc_out = tf.nn.bias_add(fcm, biases, name="loc_out")
        out_size = (shape[1], shape[2])
        assert (
            len(ff_out.shape) == 4
        )  # we can only spatial transform on a convolutional input
        h_trans = spatial_transformer.transformer(ff_out, loc_out, out_size)
        bs = ff_out.get_shape().as_list()[0]
        cs = ff_out.get_shape().as_list()[-1]
        h_trans.set_shape([bs, shape[1], shape[2], cs])
        return h_trans


def depth_preproc(
    inputs,
    l1_inpnm,
    ff_inpnm,
    node_nms,
    shape,
    spatial_op="resize",
    channel_op="concat",
    kernel_init="xavier",
    weight_decay=None,
    reuse=None,
    ksize=3,
    activation=None,
    kernel_init_kwargs=None,
):
    """Separates feedback from feedforward inputs and then combines the non feedforward inputs together"""
    ff_in, skip_ins, feedback_ins = gather_inputs(
        inputs, shape, l1_inpnm, ff_inpnm, node_nms
    )
    not_ff = feedback_ins + skip_ins
    dict_out = {"ff": None, "non_ff": None}

    if (
        len(not_ff) == 0 or ff_in is None or len(shape) != 4 or len(ff_in.shape) != 4
    ):  # we do nothing in this case, and proceed as usual (appeases initialization too)
        out_val = input_aggregator(
            inputs,
            shape,
            spatial_op,
            channel_op,
            kernel_init,
            weight_decay,
            reuse,
            ff_inpnm,
            ksize,
            activation,
            kernel_init_kwargs,
        )
        dict_out["ff"] = out_val
        return dict_out

    # aggregate non feedforward input (resize and concat by default)
    # non_ff_out = input_aggregator(not_ff, shape, spatial_op, channel_op, kernel_init, weight_decay, reuse, ff_inpnm, ksize, activation, kernel_init_kwargs)
    dict_out["ff"] = ff_in
    dict_out["non_ff"] = not_ff
    return dict_out


def gate_preproc(
    inputs,
    shape,
    spatial_op,
    channel_op,
    kernel_init,
    weight_decay,
    reuse,
    ff_inpnm,
    ksize,
    activation,
    kernel_init_kwargs,
    padding,
):
    """creates a gate for each feedback where there is a term for every feedback based on the pre memory output that the ff input is fed to
    then multiplies the gate with each feedback"""
    feedback_inps = []
    ff_inp = []
    for inp in inputs:
        if (
            inp.shape[1] == shape[1]
            and inp.shape[2] == shape[2]
            and inp.shape[3] == shape[3]
        ):
            ff_inp.append(
                inp
            )  # simply do nothing with feedforward input or inputs of the same shape
        else:
            feedback_inps.append(inp)

    out_terms = ff_inp
    for inp in feedback_inps:  # create gates for each feedback
        pat = re.compile(":|/")
        if len(inp.name.split("/")) == 1:
            orig_nm = pat.sub("__", inp.name.split("/")[-1].split("_")[0])
        else:
            orig_nm = pat.sub("__", inp.name.split("/")[-2].split("_")[0])

        nm = "gate_for_%s" % orig_nm
        with tf.variable_scope(nm, reuse=reuse):
            gate_vars = ff_inp
            for fb in feedback_inps:  # create gate variables for each feedback
                d_out = deconv(
                    fb,
                    shape=shape,
                    weight_decay=weight_decay,
                    ksize=ksize,
                    activation=activation,
                    padding=padding,
                    reuse=reuse,
                )
                gate_vars.append(d_out)
            gate_name = "gate_out_for_%s" % orig_nm
            linear_comp = tf.add_n(gate_vars, name=gate_name)
            gate_out = tf.nn.sigmoid(linear_comp)

        # transform each feedback to the right shape, then multiply by the gate
        transform_nm = "transform_for_%s" % orig_nm
        with tf.variable_scope(transform_nm, reuse=reuse):
            lin_transform_out = deconv(
                inp,
                shape=shape,
                weight_decay=weight_decay,
                ksize=ksize,
                activation=activation,
                padding=padding,
                reuse=reuse,
            )
            gate_comb = gate_out * lin_transform_out
            out_terms.append(gate_comb)

    output = tf.add_n(out_terms, name="transform_out")
    return output


def harbor(
    inputs,
    shape,
    name,
    ff_inpnm=None,
    node_nms=["split", "V1", "V2", "V4", "pIT", "aIT"],
    l1_inpnm="split",
    preproc=None,
    spatial_op="resize",
    channel_op="concat",
    kernel_init="xavier",
    kernel_init_kwargs=None,
    weight_decay=None,
    dropout=None,
    ksize=3,
    activation=None,
    padding="SAME",
    reuse=None,
    out_depth_per_input=None,
):
    """
    Default harbor function which can crop the input (as a preproc), followed by a spatial_op which by default resizes inputs to a desired shape (or pad or tile), and finished with a channel_op which by default concatenates along the channel dimension (or add or multiply based on user specification).

    :Args:
        - inputs
        - shape
    """
    if preproc == "crop":
        inputs = crop_func(
            inputs, l1_inpnm, ff_inpnm, node_nms, shape, kernel_init, channel_op, reuse
        )
    elif preproc == "depth":
        output = depth_preproc(
            inputs,
            l1_inpnm,
            ff_inpnm,
            node_nms,
            shape,
            spatial_op,
            channel_op,
            kernel_init,
            weight_decay,
            reuse,
            ksize,
            activation,
            kernel_init_kwargs,
        )
        return output
    elif preproc == "sp_transform":
        # skips and feedforward inputs were combined already and then transformed by the feedback
        output = sptransform_preproc(
            inputs,
            l1_inpnm,
            ff_inpnm,
            node_nms,
            shape,
            spatial_op,
            channel_op,
            kernel_init,
            weight_decay,
            dropout,
            reuse,
        )
        return output
    elif preproc == "gate":
        output = gate_preproc(
            inputs,
            shape,
            spatial_op,
            channel_op,
            kernel_init,
            weight_decay,
            reuse,
            ff_inpnm,
            ksize,
            activation,
            kernel_init_kwargs,
            padding,
        )
        return output

    output = input_aggregator(
        inputs,
        shape,
        spatial_op,
        channel_op,
        kernel_init,
        weight_decay,
        reuse,
        ff_inpnm,
        ksize,
        activation,
        kernel_init_kwargs,
        padding,
        out_depth_per_input=out_depth_per_input,
    )

    return output


def memory(inp, state, memory_decay=0, trainable=False, name="memory"):
    """
    Memory that decays over time
    """
    initializer = model_tool.initializer(kind="constant", value=memory_decay)

    mem = tf.get_variable(
        initializer=initializer,
        shape=1,
        dtype=tf.float32,
        trainable=trainable,
        name="memory_decay",
    )
    state = tf.add(state * mem, inp, name=name)
    return state


def residual_add(
    inp,
    res_inp,
    dtype=tf.float32,
    drop_connect_rate=None,
    kernel_init="xavier",
    kernel_init_kwargs=None,
    strides=[1, 1, 1, 1],
    padding="SAME",
    batch_norm=False,
    group_norm=False,
    num_groups=32,
    is_training=False,
    init_zero=None,
    crossdevice_bn_kwargs={},
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5,
    sp_resize=True,
    time_sep=False,
    time_suffix=None,
    bn_trainable=True,
):

    if time_sep:
        assert time_suffix is not None

    if kernel_init_kwargs is None:
        kernel_init_kwargs = {}

    if inp.shape.as_list() == res_inp.shape.as_list():
        if drop_connect_rate is not None:
            inp = drop_connect(inp, is_training, drop_connect_rate)
        return tf.add(inp, res_inp, name="residual_sum")
    elif inp.shape.as_list()[:-1] == res_inp.shape.as_list()[:-1]:
        # need to do a 1x1 conv to fix channels
        initializer = model_tool.initializer(kind=kernel_init, **kernel_init_kwargs)

        res_to_out_kernel = tf.get_variable(
            "residual_add_weights",
            [1, 1, res_inp.shape.as_list()[-1], inp.shape.as_list()[-1]],
            dtype=tf.float32,
            initializer=initializer,
        )
        projection_out = tf.nn.conv2d(
            res_inp, res_to_out_kernel, strides=strides, padding=padding
        )
        if batch_norm:
            projection_out = model_tool.batchnorm_corr(
                inputs=projection_out,
                is_training=is_training,
                decay=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                init_zero=init_zero,
                activation=None,
                data_format="channels_last",
                time_suffix=time_suffix,
                bn_trainable=bn_trainable,
                **crossdevice_bn_kwargs
            )
        elif group_norm:
            projection_out = model_tool.groupnorm(
                inputs=projection_out,
                G=num_groups,
                data_format="channels_last",
                epsilon=batch_norm_epsilon,
            )

        return tf.add(inp, projection_out)
    else:  # shape mismatch in spatial dimension
        if sp_resize:  # usually do this if strides are kept to 1 always
            res_inp = tf.image.resize_images(
                res_inp, inp.shape.as_list()[1:3], align_corners=True
            )

        initializer = model_tool.initializer(kind=kernel_init, **kernel_init_kwargs)
        res_to_out_kernel = tf.get_variable(
            "residual_add_weights",
            [1, 1, res_inp.shape.as_list()[-1], inp.shape.as_list()[-1]],
            dtype=tf.float32,
            initializer=initializer,
        )
        projection_out = tf.nn.conv2d(
            res_inp, res_to_out_kernel, strides=strides, padding=padding
        )
        if batch_norm:
            projection_out = model_tool.batchnorm_corr(
                inputs=projection_out,
                is_training=is_training,
                decay=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                init_zero=init_zero,
                activation=None,
                data_format="channels_last",
                time_suffix=time_suffix,
                bn_trainable=bn_trainable,
                **crossdevice_bn_kwargs
            )
        elif group_norm:
            projection_out = model_tool.groupnorm(
                inputs=projection_out,
                G=num_groups,
                data_format="channels_last",
                epsilon=batch_norm_epsilon,
            )

        return tf.add(inp, projection_out)


def component_conv(
    inp,
    inputs_list,
    out_depth,
    input_name=None,
    ksize=[3, 3],
    strides=[1, 1, 1, 1],
    data_format="channels_last",
    padding="SAME",
    kernel_init="xavier",
    kernel_init_kwargs=None,
    use_bias=True,
    bias=0,
    weight_decay=None,
    activation=None,
    batch_norm=False,
    group_norm=False,
    num_groups=32,
    is_training=False,
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5,
    init_zero=None,
    return_input=False,
    time_sep=False,
    time_suffix=None,
    bn_trainable=True,
    crossdevice_bn_kwargs={},
    name="component_conv",
):

    """
        Function that breaks up the convolutional kernel to its basenet and non basenet components, when given
    the name of its feedforward input. This is useful when loading basenet weights into tnn when using a
    harbor channel op of concat. Other channel ops should work with model_tool.conv just fine.
    """

    if time_sep:
        assert time_suffix is not None

    assert input_name is not None
    # assert out_shape is not None
    if weight_decay is None:
        weight_decay = 0.0
    if isinstance(ksize, int):
        ksize = [ksize, ksize]
    if kernel_init_kwargs is None:
        kernel_init_kwargs = {}
    in_depth = inp.get_shape().as_list()[-1]

    # weights
    init = model_tool.initializer(kernel_init, **kernel_init_kwargs)

    kernel_list = []
    w_idx = 0
    for input_elem in inputs_list:
        if input_name is not None and input_name in input_elem.name:
            kernel = tf.get_variable(
                initializer=init,
                shape=[
                    ksize[0],
                    ksize[1],
                    input_elem.get_shape().as_list()[-1],
                    out_depth,
                ],
                dtype=tf.float32,
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                name="weights_basenet",
            )
        else:
            kernel = tf.get_variable(
                initializer=init,
                shape=[
                    ksize[0],
                    ksize[1],
                    input_elem.get_shape().as_list()[-1],
                    out_depth,
                ],
                dtype=tf.float32,
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                name="weights_" + str(w_idx),
            )
            w_idx += 1

        kernel_list.append(kernel)

    new_kernel = tf.concat(kernel_list, axis=-2, name="weights")

    if use_bias:
        const_init = model_tool.initializer(kind="constant", value=bias)

        biases = tf.get_variable(
            initializer=const_init,
            shape=[out_depth],
            dtype=tf.float32,
            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            name="bias",
        )
    # ops
    conv = tf.nn.conv2d(inp, new_kernel, strides=strides, padding=padding)

    if use_bias:
        output = tf.nn.bias_add(conv, biases, name=name)
    else:
        output = tf.identity(conv, name=name)

    if batch_norm:
        output = model_tool.batchnorm_corr(
            inputs=output,
            is_training=is_training,
            data_format=data_format,
            decay=batch_norm_decay,
            epsilon=batch_norm_epsilon,
            init_zero=init_zero,
            activation=activation,
            time_suffix=time_suffix,
            bn_trainable=bn_trainable,
            **crossdevice_bn_kwargs
        )
    elif group_norm:
        output = model_tool.groupnorm(
            inputs=output,
            G=num_groups,
            data_format=data_format,
            epsilon=batch_norm_epsilon,
            weight_decay=weight_decay,
        )

    if activation is not None:
        output = getattr(tf.nn, activation)(output, name=activation)

    if return_input:
        return output, inp
    else:
        return output


def conv_bn(
    inp,
    out_depth,
    ksize=[3, 3],
    strides=[1, 1, 1, 1],
    data_format="channels_last",
    padding="SAME",
    kernel_init="xavier",
    kernel_init_kwargs=None,
    bias=0,
    weight_decay=None,
    activation=None,
    batch_norm=False,
    is_training=False,
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5,
    init_zero=None,
    bn_trainable=True,
    crossdevice_bn_kwargs={},
    name="conv",
):

    # assert out_shape is not None
    if weight_decay is None:
        weight_decay = 0.0
    if isinstance(ksize, int):
        ksize = [ksize, ksize]
    if kernel_init_kwargs is None:
        kernel_init_kwargs = {}
    in_depth = inp.get_shape().as_list()[-1]

    # weights
    kernel = tf.get_variable(
        initializer=model_tool.initializer(kernel_init, **kernel_init_kwargs),
        shape=[ksize[0], ksize[1], in_depth, out_depth],
        dtype=tf.float32,
        regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
        name="weights",
    )

    biases = tf.get_variable(
        initializer=model_tool.initializer(kind="constant", value=bias),
        shape=[out_depth],
        dtype=tf.float32,
        regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
        name="bias",
    )
    # ops
    conv = tf.nn.conv2d(inp, kernel, strides=strides, padding=padding)
    output = tf.nn.bias_add(conv, biases, name=name)

    if batch_norm:
        output = model_tool.batchnorm_corr(
            inputs=output,
            is_training=is_training,
            data_format=data_format,
            decay=batch_norm_decay,
            epsilon=batch_norm_epsilon,
            init_zero=init_zero,
            activation=activation,
            bn_trainable=bn_trainable,
            **crossdevice_bn_kwargs
        )

    if activation is not None:
        output = getattr(tf.nn, activation)(output, name=activation)

    return output


def spatial_fc(
    inp,
    out_depth,
    kernel_init="xavier",
    kernel_init_kwargs=None,
    bias=0.0,
    reg_scales=None,
    activation=None,
    flatten=False,
    name="spatial_fc",
):

    """
    Function that fully connects a spatial tensor of rank 4 to a flat tensor of rank 2.
    Whereas fc(inp) will flatten the input and perform an affine transformation,
    spatial_fc(inp) performs a conv op with kernel shape [H,W,D,out_depth].
    This allows for regularization that takes into account the spatial nature of the kernel.

    Args:

    inp: a rank 4 tensor with shape [Batch, H, W, D]
    out_depth: number of channels in the downstream fc layer
    reg_scales: dict with keys {weight_decay, l1, laplacian, group_sparsity}
                whose real scalar values multiply the respective regularizers. (weight_decay corresponds to L2.)
    kernel_init: in ['xavier', 'zeros', 'constant', etc.]
    kernel_init_kwargs: kwargs to pass to model_tool.initializer, e.g. 'value' for a constant init
    bias: float value for constant bias initializer
    """
    if kernel_init_kwargs is None:
        kernel_init_kwargs = {}

    # spatial dimensions of input layer
    in_shape = inp.get_shape().as_list()[1:4]

    # kernel
    reg_func = _get_regularizer(reg_scales)
    kernel = tf.get_variable(
        initializer=model_tool.initializer(kernel_init, **kernel_init_kwargs),
        shape=[in_shape[0], in_shape[1], in_shape[2], out_depth],
        dtype=tf.float32,
        regularizer=reg_func,
        name="weights",
    )

    biases = tf.get_variable(
        initializer=model_tool.initializer(kind="constant", value=bias),
        shape=[out_depth],
        dtype=tf.float32,
        regularizer=None,
        name="bias",
    )

    # ops
    # conv has full connectivity
    conv = tf.nn.conv2d(inp, kernel, strides=[1, 1, 1, 1], padding="VALID")
    output = tf.nn.bias_add(conv, biases, name=name)

    if activation is not None:
        output = getattr(tf.nn, activation)(output, name=activation)
    if flatten:
        output = tf.reshape(
            output, shape=[output.shape.as_list()[0], -1], name="flatten"
        )

    return output


def factored_fc(
    inp,
    out_depth,
    spatial_mask_init="xavier",
    spatial_mask_init_kwargs=None,
    feature_kernel_init="xavier",
    feature_kernel_init_kwargs=None,
    kernel_init=None,
    kernel_init_kwargs={},
    bias=1.0,
    spatial_reg_scales=None,
    feature_reg_scales=None,
    activation=None,
    flatten=True,
    dropout=None,
    dropout_seed=0,
    name="factored_fc",
):

    """
    Function that fully connects a spatial tensor of rank 4 to a flat tensor of rank 2.
    Whereas spatial_fc(inp) performs a conv op with kernel shape [H,W,D,out_depth],
    factored_fc(inp) performs a depth-separable conv over the H,W dimensions with a common spatial
    kernel, then takes inner product over the D dimension with a feature kernel.
    Regularizations may apply separately to the spatial mask M and the feature weights W.
    See [citation].

    Args:

    inp: a rank 4 tensor with shape [Batch, H, W, D]
    out_depth: number of channels in the downstream fc layer, i.e. num_neurons to fit N
    reg_scales: dict with keys {weight_decay, l1, laplacian, group_sparsity}
                whose real scalar values multiply the respective regularizers. (weight_decay corresponds to L2.)
    spatial_mask_init, feature_kernel_init: in ['xavier', 'zeros', 'constant', etc.]
    spatial_mask_init_kwargs, feature_kernel_init_kwargs: kwargs to pass to model_tool.initializer, e.g. 'value' for a constant init
    bias: float value for constant bias initializer
    """
    if spatial_mask_init_kwargs is None:
        spatial_mask_init_kwargs = {}
    if feature_kernel_init_kwargs is None:
        feature_kernel_init_kwargs = {}
    if kernel_init is not None:
        spatial_mask_init = kernel_init
        spatial_mask_init_kwargs = copy.deepcopy(kernel_init_kwargs)
        feature_kernel_init = kernel_init
        feature_kernel_init_kwargs = copy.deepcopy(kernel_init_kwargs)

    # spatial dimensions of input layer, H x W x D
    in_shape = inp.get_shape().as_list()[1:4]

    # spatial mask conv kernel
    reg_func = _get_regularizer(spatial_reg_scales)

    # kernel for depthwise convolution with channel_multiplier=1
    spatial_kernel = tf.get_variable(
        initializer=model_tool.initializer(
            spatial_mask_init, **spatial_mask_init_kwargs
        ),
        shape=[in_shape[0], in_shape[1], out_depth, 1],
        dtype=tf.float32,
        regularizer=reg_func,
        name="weights_spatial",
    )

    # feature kernel
    reg_func = _get_regularizer(feature_reg_scales)

    # kernel only operates in D dimension
    feature_kernel = tf.get_variable(
        initializer=model_tool.initializer(
            feature_kernel_init, **feature_kernel_init_kwargs
        ),
        shape=[in_shape[2], out_depth],
        dtype=tf.float32,
        regularizer=reg_func,
        name="weights_feature",
    )
    biases = tf.get_variable(
        initializer=model_tool.initializer(kind="constant", value=bias),
        shape=[out_depth],
        dtype=tf.float32,
        regularizer=None,
        name="bias",
    )

    # ops
    if dropout is not None:
        inp = tf.nn.dropout(inp, dropout, seed=dropout_seed, name="dropout")
    # inner product along dimension D
    inp = tf.tensordot(inp, feature_kernel, axes=[[3], [0]])  # inp now B x H x W x N

    # depthwise conv to fully connect all spatial points within a neuron
    inp = tf.nn.depthwise_conv2d(
        inp, spatial_kernel, strides=[1, 1, 1, 1], padding="VALID"
    )

    # flatten and add biases
    if flatten:
        inp = tf.squeeze(
            inp, axis=[1, 2]
        )  # do not want to accidentally squeeze N dimension if N = 1, else bias_add will throw error
    output = tf.nn.bias_add(inp, biases, name=name)

    if activation is not None:
        output = getattr(tf.nn, activation)(output, name=activation)

    return output


def shared_spatial_mlp(
    inp,
    out_depth,
    scope="shared_spatial_mlp",
    hidden_dims=[],
    bias=0.0,
    activation=tf.nn.elu,
    kernel_initializer="xavier",
    kernel_initializer_kwargs=None,
):
    """
    Applies same mlp to every every spatial feature
    """

    if kernel_initializer_kwargs is None:
        kernel_initializer_kwargs = {}
    kernel_init = model_tool.initializer(
        kind=kernel_initializer, **kernel_initializer_kwargs
    )
    bias_init = tf.constant_initializer(value=bias)

    if activation is None:
        activation = tf.identity

    assert len(inp.shape.as_list()) == 3
    B, N, F = inp.shape.as_list()
    output = tf.reshape(inp, [B * N, F], name="reshape")
    input_dim = output.shape.as_list()[1]
    mlp_dims = hidden_dims + [out_depth]

    with tf.variable_scope(scope):
        dim_now = input_dim
        num_layers = len(mlp_dims)
        for i, hidden_dim in enumerate(mlp_dims):
            kernel = tf.get_variable(
                initializer=kernel_init,
                shape=[dim_now, hidden_dim],
                dtype=tf.float32,
                name=("layer_" + str(i + 1) + "_weights"),
            )

            biases = tf.get_variable(
                initializer=bias_init,
                shape=[hidden_dim],
                dtype=tf.float32,
                name=("layer_" + str(i + 1) + "_bias"),
            )

            # ops
            output = tf.matmul(output, kernel)
            output = tf.nn.bias_add(output, biases)
            if i + 1 != num_layers:
                output = activation(output, name=("layer_" + str(i + 1) + "_output"))
                dim_now = output.shape.as_list()[1]
            else:
                output = tf.identity(output, name="mlp_output_batch")

    output = tf.reshape(output, [B, N, out_depth], name="mlp_output")

    return output


def shared_xy_graph_conv(
    inp,
    num_out_attrs,
    xy_ksize=1,
    stride=1,
    node_multiplier=1,
    kernel_init="xavier",
    kernel_init_kwargs={},
    mlp_kwargs={"hidden_dims": [256]},
    bias=0.0,
    scale=0.0,
    reshape_output=True,
):
    """
    Learns X, Y linear functions of grid position and spatially shared mappings from features to node attrs
    """

    B, H, W, C = inp.shape.as_list()
    S = stride
    init = model_tool.initializer(kind="constant", value=bias)
    bias_init = model_tool.initializer(kind="constant", value=bias)

    # X,Y coordinate functions
    ones = tf.ones(shape=[1, H, W], dtype=tf.float32)
    hw_grid = scale * tf.stack(
        [
            tf.reshape(tf.range(W, dtype=tf.float32), [1, 1, W]) * ones,
            tf.reshape(tf.range(H, dtype=tf.float32), [1, H, 1]) * ones,
        ],
        axis=-1,
    )  # [1,H,W,2] channels will be mapped to x,y

    coordinate_filter = tf.get_variable(
        "xy_filter", shape=[1, 1, 2, 1], dtype=tf.float32, initializer=init
    )
    coordinate_filter *= tf.reshape(
        tf.constant([1.0, -1.0], dtype=tf.float32), [1, 1, 2, 1]
    )
    coordinate_bias = tf.get_variable(
        "xy_bias", shape=[1, 1, 1, 2], dtype=tf.float32, initializer=bias_init
    )

    xy_grid = tf.nn.depthwise_conv2d(
        hw_grid,
        coordinate_filter,
        strides=[1, S, S, 1],
        padding="SAME",
        name="xy_linear",
    )
    xy_grid += (
        coordinate_bias  # [1,H//S,W//S,2] where first channel are X, second are Y
    )

    # larger spatial scale coordinate learning
    if xy_ksize > 1:
        xfilter = tf.get_variable(
            "x_filter", shape=[1, xy_ksize, C, 2], dtype=tf.float32, initializer=init
        )
        yfilter = tf.get_variable(
            "y_filter", shape=[xy_ksize, 1, C, 2], dtype=tf.float32, initializer=init
        )

        dx, dzx = tf.split(
            tf.nn.conv2d(inp, xfilter, strides=[1, S, S, 1], padding="SAME"),
            [1, 1],
            axis=-1,
        )  # [B,H//S,W//S,2]
        dy, dzy = tf.split(
            tf.nn.conv2d(inp, yfilter, strides=[1, S, S, 1], padding="SAME"),
            [1, 1],
            axis=-1,
        )  # [B,H//S,W//S,2]
        dz = dzx + dzy

        xy_grid += tf.concat([dx, dy], axis=3)
    else:
        xy_grid = tf.tile(xy_grid, [B, 1, 1, 1])
        dz = tf.zeros([B, H // S, W // S, 1], dtype=tf.float32)

    # graph conv where nodes are the spatial features of the input
    if S > 1:
        downsample_kernel = tf.get_variable(
            "downsample_kernel", shape=[S, S, C, 1], dtype=tf.float32, initializer=init
        )
        out = tf.nn.depthwise_conv2d(
            inp,
            downsample_kernel,
            strides=[1, S, S, 1],
            padding="SAME",
            name="downsample_conv",
        )
        out = tf.reshape(out, [B, (H * W) // (S**2), C])
    else:
        out = tf.reshape(inp, [B, H * W, C])

    out = shared_spatial_mlp(
        out, out_depth=(num_out_attrs * node_multiplier), **mlp_kwargs
    )
    out = tf.reshape(out, [B, H // S, W // S, num_out_attrs, node_multiplier])

    # add coordinates to MLP outputs
    out += tf.expand_dims(
        tf.concat(
            [
                xy_grid,
                dz,
                tf.zeros([B, H // S, W // S, num_out_attrs - 3], dtype=tf.float32),
            ],
            axis=3,
        ),
        -1,
    )  # [B,H,W,num_out_attrs,node_multiplier]

    # reshape to 4-tensor with spatial dimensions combined
    out = tf.transpose(out, [0, 1, 2, 4, 3])  # [B,H//S,W//S,M,num_out_attrs]
    if reshape_output:
        out = tf.reshape(
            out, [B, 1, (H * W * node_multiplier) // (S**2), num_out_attrs]
        )

    return out


def drop_connect(inputs, is_training, drop_connect_rate):
    if (not is_training) or (
        drop_connect_rate == 1.0
    ):  # if drop_connect_rate is 1.0 we do not apply it
        return inputs

    print("applying drop connect with rate %.2f" % drop_connect_rate)

    # compute keep prob
    keep_prob = drop_connect_rate

    # compute drop_connect_tensor
    batch_size = tf.shape(inputs)[0]
    random_tensor = keep_prob
    random_tensor = random_tensor + tf.random_uniform(
        [batch_size, 1, 1, 1], dtype=inputs.dtype
    )
    binary_tensor = tf.floor(random_tensor)
    output = tf.div(inputs, keep_prob) * binary_tensor
    return output


def squeeze_and_excitation(
    inputs,
    reduced_channels,
    activation=tf.nn.relu,
    kernel_init="variance_scaling",
    kernel_init_kwargs={"seed": 0},
):
    """
    Squeeze and Excitation Layer

    inputs: [B,H,W,C] conv2d tensor with channels last
    reduction_ratio: float in [0,1] to determine how much to squeeze channels
    activation: nonlinear function to apply after reduction conv
    """
    B, H, W, C = inputs.shape.as_list()
    rC = max(1, int(reduced_channels))

    se_tensor = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)  # [B,1,1,C]
    with tf.variable_scope("se_reduce"):
        se_tensor = model_tool.conv(
            se_tensor,
            rC,
            ksize=[1, 1],
            use_bias=True,
            activation=activation,
            kernel_init=kernel_init,
            kernel_init_kwargs=kernel_init_kwargs,
        )
    with tf.variable_scope("se_expand"):
        se_tensor = model_tool.conv(
            se_tensor,
            C,
            ksize=[1, 1],
            use_bias=True,
            activation=None,
            kernel_init=kernel_init,
            kernel_init_kwargs=kernel_init_kwargs,
        )

    print("squeeze-excitation with %d to %d channels" % (C, rC))
    return tf.nn.sigmoid(se_tensor) * inputs
