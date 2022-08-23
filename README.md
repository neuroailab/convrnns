# Convolutional Recurrent Neural Networks (ConvRNNs) Model Zoo
This repo contains ConvRNNs of varying depths that are pre-trained on ImageNet categorization, ready to be used for inference and comparison to neural & behavioral data.
Training these models tends to take considerable compute resources, and our aim is to make inference as accessible as possible, with minimal package dependencies.

## What are ConvRNNs?
ConvRNNs are neural networks that have a combination of layer-local recurrent circuits implanted into a feedforward convolutional neural network (CNN), along with the potential addition of long-range feedback connections between layers.
Feedforward CNNs are therefore a special case of ConvRNNs, and we consider ConvRNNs of varying feedforward depths: "shallow" (6 layers) and "intermediate" (11 layers).
All models are "biologically unrolled", where propagation along each arrow takes one time step (~10 *ms*) to mimic conduction delays between cortical layers in the visual system.

<img width="1626" alt="Repo_Vis" src="https://user-images.githubusercontent.com/12835914/154434249-a29a2e2d-c448-4666-bb21-c9d8db4a6d33.png">

## Available Models
Models are named according to the convention of `[cell type]_[feedforward depth]_[optional decoder/additional properties]`, all of which are trained on ImageNet categorization and described in [our paper](https://www.biorxiv.org/content/10.1101/2021.02.17.431717).
Some models may be better suited than others based on your needs, but we recommend: 
- `rgc_intermediate`, if you want a model that is task performant *and* well matches temporally varying primate core object recognition data (neural & behavioral recordings). Note that this model and the other `rgc_intermediate*` models are the only models that also have long-range feedback connections (found through large-scale meta-optimization), in addition to parameter efficient layer-local recurrent cells.
- `ugrnn_intermediate`, if you want a model that *best* matches temporally varying primate data.
- `rgc_intermediate_t22_tmaxconf`, if you want a model that is task performant *and unrolled for a longer amount of timesteps (22) during training*. This model and the `ugrnn_intermediate_t30*` models used a constant image presentation (unlike the other models where the image presentation sequence turns off eventually).
- `rgc_shallow`, if you want a shallow (6 layer) model that is most task performant and best matches temporally varying primate data (when compared to other shallow ConvRNNs). Use this model if you have a resource limitation when running inference or the domain of application only feasibly allows the use of shallow networks. Otherwise, we generally recommend using any of the above intermediate (11 layer) models.

Below is a table of all the models and their ImageNet validation set accuracy:

| Model Name      | Top1 Accuracy | Top5 Accuracy |
| --------------- | ------------- | ------------- |
| `timedecay_shallow`      | 50.06%       | 72.97%       |
| `simplernn_shallow`      | 61.11%       | 83.15%       |
| `gru_shallow`      | 62.16%       | 84.02%       |
| `lstm_shallow`      | 63.43%       | 84.89%       |
| `intersectionrnn_shallow`      | 65.42%       | 86.02%       |
| `ugrnn_shallow`      | 60.82%       | 83.23%       |
| `rgc_shallow`      | 66.97%       | 87.02%       |
| `timedecay_intermediate`      | 67.52%       | 87.80%       |
| `simplernn_intermediate`      | 70.47%       | 89.48%       |
| `gru_intermediate`      | 69.81%       | 89.25%       |
| `lstm_intermediate`      | 69.57%       | 89.18%       |
| `intersectionrnn_intermediate`      | 68.98%      | 88.67%       |
| `ugrnn_intermediate`      | 69.78%       | 89.21%       |
| `ugrnn_intermediate_t30`      | 70.09%       | 89.29%       |
| `ugrnn_intermediate_t30_dthresh`      | 69.30%       | 88.97%       |
| `ugrnn_intermediate_t30_tmaxconf`      | 70.04%       | 89.29%       |
| `rgc_intermediate`      | 73%       | 91.16%       |
| `rgc_intermediate_dthresh`      | 71.81%       | 90.16%       |
| `rgc_intermediate_tmaxconf`      | 72.29%       | 90.52%       |
| `rgc_intermediate_t22_tmaxconf`      | 72.98%       | 90.80%       |

## Getting Started
It is recommended that you install this repo within a virtual environment (both Python 2.7 and 3.x are supported), and run inferences there.
An example command for doing this with `anaconda` would be:
```
conda create -y -n your_env python=3.6.10 anaconda
```
To install this package and all of its dependecies, clone this repo on your machine and then install it via pip:
```
git clone https://github.com/neuroailab/convrnns
cd convrnns/
pip install -e .
```
Note that CPU is used by default (having a GPU is not required, though recommended if running inferences on a large number of images).
If you prefer to use GPU, then be sure to install `tensorflow-gpu==1.13.1` instead of `tensorflow`, and ensure that you have `CUDA 10.0` and `cudnn 7.3` installed.

Next, download the model checkpoints by running:
```
./get_checkpoints.sh
```
The total size is currently 5.3 GB, so if you prefer to download only the weights of one particular model or just a few, you can modify the script to do that.

## Extracting Model Features
### i. Model Layers
The model layers for the `shallow` ConvRNNs are named: `'conv1','conv2','conv3','conv4','conv5','conv6','imnetds'`.
The ConvRNN cells are embedded in layers `'conv3'`, `'conv4'`, and `'conv5'` for these models.

The model layers for the `intermediate` ConvRNNs are named: `'conv1','conv2','conv3','conv4','conv5','conv6','conv7','conv8','conv9','conv10','imnetds'`.
The ConvRNN cells are embedded in layers `'conv4'` through `'conv10'` for these models.

For both model types, `imnetds` is always the final 1000-way categorization layer.

Here is an example for extracting the features of `'conv9'` and `'conv10'`:
```
python run_model.py --model_name='rgc_intermediate' --out_layers='conv9,conv10'
```
The above command returns a dictionary whose keys are the model layers, and whose values are the features for each timepoint (starting from when that layer has a feedforward output).
**Note:** you will need to supply your own images by modifying the `run_model.py` script prior to running it.

If you are interested in neural fits, we generally recommend re-fitting the model features based on your neural data and transform class that you are using, and picking the model layer(s) that yield(s) maximum neural predictivity for that visual area.
For reference, for our data, we have found that for the `shallow` ConvRNNs, model layer `'conv3'` best matches to V4, `'conv4'` best matches to pIT, and `'conv5'` best matches to cIT/aIT.
For the `intermediate` ConvRNNs, model layers `'conv5'` and `'conv6'` best match to V4, `'conv7'` and `'conv8'` best match to pIT, and `'conv9'` and `'conv10'` best match to cIT/aIT.

### ii. Stimulus Presentation
The models all expect images of size 224x224x3, normalized between 0 and 1, and by the ImageNet mean and std (we include code that performs this normalization automatically in `run_model.py`).
When you compare to neural data, it is strongly recommended to **present the images in the same way they were presented to the subjects**, where each model timestep roughly corresponds to 10ms.
For example, in our work, the images were presented to primates for 260 ms, and "turned off" (replaced with a mean gray stimulus) at 100 ms.
Therefore, we unrolled the models for 26 timesteps, with `image_off` set to 10.

For ImageNet performance reporting in the table above, by default (`image_pres='default'`), the models are automatically unrolled (16 timesteps for `shallow` and 17 timesteps for `intermediate`), and given the same video presentation format as they were *trained* with, which involves for all models to shut off the image presentation with a mean gray stimulus after 12 timesteps (however, the models with `t22` and `t30` in their names were trained with a *constant* image presentation for 22 and 30 timesteps, respectively).
If you prefer to unroll the models for a different number of timesteps and with a constant image presentation, then specify a value for `times` and `image_pres='constant'`.
If you prefer to have the model image presentation to turn off at a specified point, then specify a value for `image_off`, and replace mean gray (0.5) with a different off stimulus if that is different in your experimental setup.

## Cite
If you used this codebase for your research, please consider citing our paper:
```
@article{Nayebi2022ConvRNNs,
  title={Recurrent Connections in the Primate Ventral Visual Stream Mediate a Tradeoff Between Task Performance and Network Size During Core Object Recognition},
  author={Nayebi, Aran and Sagastuy-Brena, Javier and Bear, Daniel M and Kar, Kohitij and Kubilius, Jonas and Ganguli, Surya and Sussillo, David and DiCarlo, James J and Yamins, Daniel LK},
  journal={Neural Computation},
  volume={34},
  pages={1652--1675},
  year={2022},
  publisher={MIT Press}
}
```

## Contact
If you have any questions or encounter issues, either submit a Github issue here or email `anayebi@stanford.edu`.
