#!/bin/bash

base_url=https://convrnnckpts.s3-us-west-1.amazonaws.com

for model in rgc_intermediate rgc_intermediate_tmaxconf rgc_intermediate_t22_tmaxconf rgc_intermediate_dthresh ugrnn_intermediate intersectionrnn_intermediate lstm_intermediate gru_intermediate simplernn_intermediate timedecay_intermediate ugrnn_intermediate_t30 ugrnn_intermediate_t30_tmaxconf ugrnn_intermediate_t30_dthresh ugrnn_shallow rgc_shallow intersectionrnn_shallow lstm_shallow gru_shallow simplernn_shallow timedecay_shallow
do
    mkdir -p ./ckpts/${model}/
    curl -fLo ./ckpts/${model}/model.ckpt.data-00000-of-00001 ${base_url}/${model}/model.ckpt.data-00000-of-00001
    curl -fLo ./ckpts/${model}/model.ckpt.index ${base_url}/${model}/model.ckpt.index
    curl -fLo ./ckpts/${model}/model.ckpt.meta ${base_url}/${model}/model.ckpt.meta
done
