import os
import tensorflow as tf
import numpy as np
from convrnns.utils.loader import get_restore_vars, MODEL_TO_KWARGS
from convrnns.models.model_func import model_func


def normalize_ims(x):
    # ensures that images are between 0 and 1
    x = x.astype(np.float32)
    assert np.amin(x) >= 0
    if np.amax(x) > 1:
        assert np.amax(x) <= 255
        print("Normalizing images to be between 0 and 1")
        x /= 255.0

    # this is important to preserve on new stimuli, since the models were trained with these image normalizations
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - imagenet_mean) / imagenet_std
    return x


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Which model to load.",
        required=True,
    )
    parser.add_argument(
        "--out_layers",
        type=str,
        default="imnetds",
        help="Which layers of the model to output. Pass in a comma separated list if you want multiple layers.",
    )
    parser.add_argument(
        "--gpu", type=str, default=None, help="Which gpu to use. Default is to use CPU."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./ckpts",
        help="Path to directory where checkpoints will be stored.",
    )
    parser.add_argument(
        "--image_pres",
        type=str,
        default="default",
        choices=["default", "constant", "neural"],
        help="The type of image presentation to use. Default is to use what was used during training. Constant means the image presentation is copied times timesteps. Neural is the image presentation used for comparing to neural and behavioral data over 260 ms.",
    )
    parser.add_argument(
        "--times",
        type=int,
        default=None,
        help="Number of timesteps to unroll the model. If None, default is to use what was used during training.",
    )
    parser.add_argument(
        "--image_off",
        type=int,
        default=None,
        help="Number of timesteps to turn off image presentation if image_pres != constant. If None, default is to use what was used during training.",
    )
    parser.add_argument(
        "--include_all_times",
        type=bool,
        default=False,
        help="Whether to include all timepoints of the model. Default is to include the outputs starting when that layer has a feedforward output (recommended).",
    )
    parser.add_argument(
        "--include_logits",
        type=bool,
        default=False,
        help="Whether to include ImageNet categorization logits.",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        # If GPUs available, select which to evaluate on
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    out_layers = args.out_layers.split(",")

    # x would be the actual images you want to run through the model
    num_imgs = 5
    x = np.zeros((num_imgs, 224, 224, 3))
    x = normalize_ims(x)
    inputs = tf.placeholder(tf.float32, shape=[num_imgs, 224, 224, 3])
    model_kwargs = MODEL_TO_KWARGS[args.model_name]
    y = model_func(
        inputs=inputs,
        out_layers=out_layers,
        image_pres=args.image_pres,
        times=args.times,
        image_off=args.image_off,
        include_all_times=args.include_all_times,
        include_logits=args.include_logits,
        **model_kwargs
    )
    sess = tf.Session()
    CKPT_PATH = os.path.join(args.ckpt_dir, "{}/model.ckpt".format(args.model_name))
    restore_vars = get_restore_vars(CKPT_PATH)
    tf.train.Saver(var_list=restore_vars).restore(sess, CKPT_PATH)
    y_eval = sess.run(y, feed_dict={inputs: x})
    print(y_eval)
    sess.close()
