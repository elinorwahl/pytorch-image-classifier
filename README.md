# PyTorch Image Classifier

PyTorch is an open-source machine learning library for the Python language, created by Facebook’s artificial intelligence research team, and one of the many uses it can be put to is image recognition software. This project’s purpose is to use PyTorch to classify images of flowers common to the United Kingdom, using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html), the `torchvision` package, and a pre-trained neural network. It consists of two parts - a Jupyter notebook, and two Python files adapted for use in Command Line, with user interface tools that give the user a range of options for the network architecture.

## Description

This project consists of two distinct parts, with similar capabilities:

- A Jupyter notebook, `pytorch_image_classifier.ipynb`, which loads and processes image data, trains a pre-trained network on the new data, saves the updated model, then loads it for use in classifying new images. 
- Two Python programs, `train.py` and `predict.py`, which perform the same functions as the notebook using command line prompts. `train.py` also adds options for the user to modify the network architecture without changing the code.

This is adapted from a project available through Udacity’s Data Scientist course materials, which can be found [here](https://github.com/udacity/DSND_Term1/tree/master/projects/p2_image_classifier).

## Installation

To allow the network to train at a reasonable speed, you will have to enable GPU performance. Both the notebook and command line programs are designed to run without a GPU, if necessary, but the training will take much more time, depending on your device and internet speed. If your device has an Nvidia or ATI/AMD Radeon graphics card, it has dedicated GPU. If not, some alternatives for web-hosted GPU use are [Amazon AWS](https://aws.amazon.com/hpc/) and [Crestle](https://www.crestle.com).

## Usage

If you're using the Python files in the command line, there are a range of arguments that can be input before the programs are executed.

For `train.py`:

- `data-dir`, the path to the image folder you intend to train the network on.
- `--save-dir`, the path to the save checkpoint.
- `--arch`, the choice of pre-trained network architecture, which can be `densenet`, `resnet`, or `vgg`; the default is DenseNet.
- `--hidden_units`, the number of hidden units within the network; the default number is 512.
- `--learning_rate`, the rate at which the network learns; the default number is 0.003.
- `--epochs`, the number of epochs that the network runs its training function for; the default number is 10.
- `--gpu`, a Boolean True/False argument that determines whether or not GPU is enabled for training; the default setting is True.

For `predict.py`:

- `input`, the path to the image file that the trained network will make predictions on.
- `checkpoint`, the path to the trained classifier checkpoint to be loaded.
- `--top_k`, the number of top most-likely classes of predictions to be returned; the default number is 5.
- `--category_names`, the path to the file that provides image labels for the predicted categories; the default is `cat_to_name.json`.
- `--gpu`, a Boolean True/False argument that determines whether or not GPU is enabled for training; the default setting is True.
