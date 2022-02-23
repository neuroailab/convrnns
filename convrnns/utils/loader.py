import tensorflow as tf


def get_restore_vars(save_file):
    """Create the `var_list` init argument to tf.Saver from save_file.
    Extracts the subset of variables from tf.global_variables that match the
    name and shape of variables saved in the checkpoint file, and returns these
    as a list of variables to restore.
    To support multi-model training, a model prefix is prepended to all
    tf global_variable names, although this prefix is stripped from
    all variables before they are saved to a checkpoint. Thus,
    Args:
        save_file: path of tf.train.Saver checkpoint.
    Returns:
        dict: checkpoint variables.
    """
    reader = tf.train.NewCheckpointReader(save_file)
    var_shapes = reader.get_variable_to_shape_map()

    # Specify which vars are to be restored vs. reinitialized.
    all_vars = {v.op.name: v for v in tf.global_variables()}
    restore_vars = {name: var for name, var in all_vars.items() if name in var_shapes}

    # These variables are stored in the checkpoint,
    # but do not appear in the current graph
    in_ckpt_not_in_graph = [
        name
        for name in var_shapes.keys()
        if (name not in all_vars.keys())
        and (not name.endswith("/Momentum"))
        and (name != "global_step")
    ]
    if len(in_ckpt_not_in_graph) > 0:
        print("Vars in ckpt, not in graph:\n" + str(in_ckpt_not_in_graph))
        raise ValueError

    # Ensure the vars to restored have the correct shape.
    var_list = {}

    for name, var in restore_vars.items():
        var_shape = var.get_shape().as_list()
        if var_shape == var_shapes[name]:
            var_list[name] = var
        else:
            print(
                "Shape mismatch for %s" % name + str(var_shape) + str(var_shapes[name])
            )
    return var_list


MODEL_TO_KWARGS = {}
for cell in [
    "timedecay",
    "simplernn",
    "lstm",
    "gru",
    "intersectionrnn",
    "ugrnn",
    "rgc",
]:
    for depth in ["shallow", "intermediate"]:
        model_name = cell + "_" + depth
        MODEL_TO_KWARGS[model_name] = {"base_name": "convrnns/models/{}".format(model_name)}
MODEL_TO_KWARGS["rgc_intermediate_tmaxconf"] = {
    "base_name": "convrnns/models/rgc_intermediate",
    "decoder_type": "max_conf",
    "decoder_trainable": True,
}
MODEL_TO_KWARGS["rgc_intermediate_t22_tmaxconf"] = {
    "base_name": "convrnns/models/rgc_intermediate_t22",
    "decoder_type": "max_conf",
    "decoder_trainable": True,
}
MODEL_TO_KWARGS["rgc_intermediate_dthresh"] = {
    "base_name": "convrnns/models/rgc_intermediate",
    "decoder_type": "thresh",
    "decoder_trainable": False,
}
MODEL_TO_KWARGS["ugrnn_intermediate_t30"] = {
    "base_name": "convrnns/models/ugrnn_intermediate_t30"
}
MODEL_TO_KWARGS["ugrnn_intermediate_t30_tmaxconf"] = {
    "base_name": "convrnns/models/ugrnn_intermediate_t30",
    "decoder_type": "max_conf",
    "decoder_trainable": True,
}
MODEL_TO_KWARGS["ugrnn_intermediate_t30_dthresh"] = {
    "base_name": "convrnns/models/ugrnn_intermediate_t30",
    "decoder_type": "thresh",
    "decoder_trainable": False,
}
