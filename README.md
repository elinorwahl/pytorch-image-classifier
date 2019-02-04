# PyTorch Image Classifier

PyTorch is an open-source machine learning library for the Python language, created by Facebook’s artificial intelligence research team. One of the uses it can be put to is image recognition. This project’s purpose is to use PyTorch to classify images of flowers common to the United Kingdom, using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html), the `torchvision` package, and a pre-trained neural network. It consists of two distinct parts - a Jupyter notebook, and two Python files adapted for use in Command Line, with user interface tools that give the user a range of options for the network architecture.

## Description

This project consists of two distinct parts, with similar capabilities:

- A Jupyter notebook, `pytorch_image_classifier.ipynb`, which loads and processes image data, trains a pre-trained network on the new data, saves the updated model, then loads it for use in classifying new images. 
- Two Python programs, `train.py` and `predict.py`, which perform the same functions as the notebook using command line prompts. `train.py` also adds options for the user to modify the network architecture without changing the code.

This is adapted from a project available through Udacity’s Data Scientist course materials, which can be found [here](https://github.com/udacity/DSND_Term1/tree/master/projects/p2_image_classifier).
