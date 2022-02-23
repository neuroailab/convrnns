import os, sys, copy
import tensorflow as tf
import numpy as np
from collections import OrderedDict

from convrnns.utils import main
from convrnns.utils.decoders import temporal_decoder
from convrnns.utils.cells import (
    tnn_ReciprocalGateCell,
    tnn_ConvNormBasicCell,
    tnn_ConvLSTMCell,
    tnn_ConvUGRNNCell,
    tnn_ConvIntersectionRNNCell,
    tnn_ConvGRUCell,
)
from convrnns.utils.median_rgcell import tnn_median_ReciprocalGateCell
from convrnns.configs.median_rgcell_cfg import median_config_dict, edges_5


def str_to_cell(convrnn_type, num_layers):
    if convrnn_type.lower() == "rgc":
        if num_layers == 11:
            # for preserving scoping when loading checkpoint
            return tnn_median_ReciprocalGateCell
        else:
            return tnn_ReciprocalGateCell
    elif convrnn_type.lower() == "lstm":
        return tnn_ConvLSTMCell
    elif convrnn_type.lower() == "gru":
        return tnn_ConvGRUCell
    elif convrnn_type.lower() == "simplernn":
        return tnn_ConvNormBasicCell
    elif convrnn_type.lower() == "ugrnn":
        return tnn_ConvUGRNNCell
    elif convrnn_type.lower() == "intersectionrnn":
        return tnn_ConvIntersectionRNNCell
    else:
        raise ValueError


def make_movie(ims, times, image_off):
    blank = tf.constant(value=0.5, shape=ims.get_shape().as_list(), name="split")
    pres = ([ims] * image_off) + ([blank] * (times - image_off))
    return pres


def set_imagepres(
    ims, image_pres, num_layers, long_unroll=None, times=None, image_off=None
):

    if image_off is None:
        if image_pres in ["default", "neural"]:
            if image_pres == "default":
                image_off = 12
                assert times is None
                if num_layers == 11:
                    times = 17
                    if long_unroll is not None:
                        times = long_unroll
                elif num_layers == 6:
                    times = 16
                else:
                    raise ValueError
            else:
                image_off = 10
                times = 26
            if (long_unroll is not None) and (image_pres == "default"):
                # the long unroll models are trained with constant images
                pres = ims
            else:
                pres = make_movie(ims=ims, times=times, image_off=image_off)
        else:
            assert image_pres == "constant"
            assert times is not None
            pres = ims
    else:
        # customize your own presentation
        assert image_pres != "constant"
        assert times is not None
        pres = make_movie(ims=ims, times=times, image_off=image_off)
    print(
        "Image Presentation Type: {}, Times: {}, Image Off Timestep: {}".format(
            image_pres, times, image_off
        )
    )
    return pres, times, image_off


def model_func(
    inputs,
    base_name,
    convrnn_type=None,
    out_layers="imnetds",
    image_pres="default",
    times=None,
    image_off=None,
    include_logits=False,
    include_all_times=False,
    decoder_type="last",
    decoder_trainable=False,
    decoder_start=None,
    decoder_end=None,
):

    if not base_name.endswith(".json"):
        base_name += ".json"
    print("Using base: {}".format(base_name))

    G = main.graph_from_json(base_name)
    if convrnn_type is None:
        convrnn_type = base_name.split("/")[-1].split("_")[0]
    convrnn_type = convrnn_type.lower()
    print("Using {} convrnn cell".format(convrnn_type))
    num_layers = len(G)
    assert num_layers in [6, 11]
    layer_params = {}
    layer_start_time = {}
    for l in range(1, num_layers):
        layer_params["conv{}".format(l)] = {}
        layer_start_time["conv{}".format(l)] = l
    layer_params["imnetds"] = {}
    layer_start_time["imnetds"] = num_layers

    if num_layers == 11:
        cell_layers = ["conv{}".format(l) for l in range(4, num_layers)]
        if convrnn_type == "rgc":
            # add median rgc parameters
            cell_params = copy.deepcopy(
                median_config_dict["model_params"]["cell_params"]
            )
            # add feedback specific params
            cell_params["feedback_activation"] = tf.identity
            cell_params["feedback_entry"] = "out"
            cell_params["feedback_depth_separable"] = False
            cell_params["feedback_filter_size"] = 1
            for k in cell_layers:
                layer_params[k]["cell_params"] = cell_params.copy()
    elif num_layers == 6:
        cell_layers = ["conv{}".format(l) for l in range(3, num_layers)]
        if convrnn_type == "lstm":
            layer_params = np.load("./convrnns/configs/lstm_shallow.npz", allow_pickle=True)[
                "arr_0"
            ][()]["model_params"]["layer_params"]
    if not isinstance(out_layers, list):
        out_layers = [out_layers]

    # set up image presentation, note that inputs is a tensor now
    ims = tf.identity(inputs, name="split")
    long_unroll = None
    if "t30" in base_name:
        long_unroll = 30
    elif "t22" in base_name:
        long_unroll = 22

    pres, times, image_off = set_imagepres(
        ims=ims,
        image_pres=image_pres,
        num_layers=num_layers,
        long_unroll=long_unroll,
        times=times,
        image_off=image_off,
    )

    # graph building stage
    with tf.variable_scope("tnn_model"):
        for node, attr in G.nodes(data=True):
            # explicitly assign non-default convrnn cells
            this_layer_params = layer_params[node]
            if (node in cell_layers) and (convrnn_type != "timedecay"):
                attr["cell"] = str_to_cell(
                    convrnn_type=convrnn_type, num_layers=num_layers
                )
                curr_cell_params = this_layer_params.get("cell_params", {}).copy()
                assert curr_cell_params is not None
                for k, v in curr_cell_params.items():
                    attr["kwargs"]["memory"][1][k] = v

        # add non feedforward edges for rgc_intermediate
        if (convrnn_type == "rgc") and (num_layers == 11):
            G.add_edges_from(edges_5)

        # initialize graph structure
        main.init_nodes(
            G, input_nodes=["conv1"], batch_size=ims.get_shape().as_list()[0]
        )

        print("Unrolling model for {} timesteps".format(times))
        # unroll graph
        main.unroll(G, input_seq={"conv1": pres}, ntimes=times)

        # collect last timestep output
        if decoder_type == "last":
            logits = G.node["imnetds"]["outputs"][-1]
            logits = tf.squeeze(logits)
        else:
            if decoder_start is None:
                decoder_start = num_layers
            if decoder_end is None:
                decoder_end = times
            logits_list = [
                G.node["imnetds"]["outputs"][t]
                for t in range(decoder_start, decoder_end)
            ]

            logits = temporal_decoder(
                logits_list, name=decoder_type, trainable=decoder_trainable
            )

        assert len(logits.shape) == 2

    outputs = {}
    if include_logits:
        outputs["logits"] = logits
    for curr_layer in out_layers:
        outputs[curr_layer] = OrderedDict()
        if include_all_times:
            curr_start = 0
        else:
            # return the outputs starting when that layer has a feedforward output
            curr_start = layer_start_time[curr_layer]
        for t in range(curr_start, times):
            outputs[curr_layer][t] = G.node[curr_layer]["outputs"][t]
    return outputs
