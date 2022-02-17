import tensorflow as tf
from collections import OrderedDict

edges_5 = [
    ("conv8", "conv5"),
    ("conv9", "conv6"),
    ("conv10", "conv7"),
    ("conv7", "conv6"),
    ("conv10", "conv9"),
]

median_config_dict = {
    "model_params": {
        "cell_params": OrderedDict(
            [
                ("tau_depth_separable", True),
                ("tau_filter_size", 7.0),
                ("residual_to_out_gate", False),
                ("feedback_activation", tf.nn.elu),
                ("residual_to_cell_gate", False),
                ("out_residual", True),
                ("feedback_depth_separable", True),
                ("tau_nonlinearity", tf.nn.sigmoid),
                ("ff_depth_separable", False),
                ("gate_nonlinearity", tf.nn.tanh),
                ("cell_to_out", False),
                ("input_to_cell", True),
                ("cell_residual", False),
                ("in_out_depth_separable", False),
                ("layer_norm", False),
                ("input_activation", tf.nn.elu),
                ("ff_filter_size", 2.0),
                ("feedback_filter_size", 8.0),
                ("feedback_entry", "input"),
                ("input_to_out", True),
                ("in_out_filter_size", 3.0),
                ("tau_offset", 0.9219348064291611),
                ("tau_multiplier", -0.9219348064291611),
                ("tau_bias", 4.147336708899556),
                ("out_activation", tf.nn.elu),
                ("weight_decay", 0.0002033999204146308),
                ("kernel_initializer", "variance_scaling"),
                ("kernel_initializer_kwargs", {"scale": 0.6393378386273998}),
                ("cell_depth", 64),
                ("gate_depth_separable", True),
                ("gate_filter_size", 7.0),
                ("gate_offset", 0.7006566684988862),
                ("gate_multiplier", -0.7006566684988862),
                ("gate_bias", 2.776542926439013),
                ("cell_activation", tf.nn.elu),
            ]
        ),
        "image_off": 12,
        "times": 17,
    }
}
