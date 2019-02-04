
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image
# from train import model_arch

def get_input_args():
    ''' Retrieve and parse the command line arguments created and defined using the argparse module. '''
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='image to load and make predictions on')
    parser.add_argument('checkpoint', type=str, help='load trained classifier from checkpoint')
    parser.add_argument('--top_k', default=5, type=int, help='return top K most likely classes')
    parser.add_argument('--category_names', default='cat_to_name.json', type=str, help='return names of predicted categories')
    parser.add_argument('--gpu', default=False, action='store_true', help='use GPU for inference')
    
    return parser.parse_args()


def load_checkpoint(filepath, gpu):
    ''' Loads a checkpoint and rebuilds a pre-trained model. '''
    # Enable choice of CPU or GPU functioning
    if gpu and torch.cuda.is_available():
        loaded_model = torch.load(filepath)
    else:
        loaded_model = torch.load(filepath, map_location=lambda storage, loc: storage)
        
    # pretrained_model, input_size = model_arch(loaded_model['arch'])
    pretrained_model.arch = loaded_model['arch']
    pretrained_model.class_idx = loaded_model['class_to_idx']
    pretrained_model.classifier = loaded_model['classifier']
    pretrained_model.load_state_dict(loaded_model['state_dict'])

    # Enables GPU functioning, when available
    if gpu and torch.cuda.is_available():
        pretrained_model.cuda()

    return pretrained_model, pretrained_model.class_idx


def process_image(image):
    ''' Scale, crop, and normalize a PIL image for a PyTorch model,
        and return the data in an Numpy array. '''
	# Resizes the image and makes the shorter side 256 pixels
    width, height = image.size
    ratio = width / height
    if width < height:
        width = 256
        height = width / ratio
    elif width > height:
        height = 256
        width = ratio * height
    else:
        width, height = 256
    image = image.resize((round(width), round(height)))

    # Crops the image and converts to an nparray
    np_image = np.array(image.crop((16, 16, 240, 240)))

    # Normalizes the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image / 255 - mean) / std

    # Changes the color channel to meet PyTorch expectations
    return np_image.transpose(2, 0, 1)


def predict(image_path, model, class_idx, top_k, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model. '''
    model.eval()
    # Implement the code to predict the class from an image file
    image = Image.open(image_path)
    image_tensor = torch.from_numpy(process_image(image))
    image_tensor.unsqueeze_(0)
    model = model.double()

    # Swap the dictionary keys and values to obtain the numerical classes of the flower names
    idx_to_class = {v: k for k, v in model.class_idx.items()}

    # Enable GPU functioning for prediction, when available
    if gpu and torch.cuda.is_available():
        model.cuda()
        image_tensor = image_tensor.cuda()
    
    with torch.no_grad():
        output = F.log_softmax(model(image_tensor), dim=1)
        prediction = torch.exp(output).data.topk(top_k)
    probs, classes = prediction[0].tolist()[0], list(map(lambda i:idx_to_class[i], prediction[1].tolist()[0]))

    return probs, classes


def predict_names(classes, category_names):
    ''' Map the category (or categories) of an image to the category names. '''
    # Load a mapping from category label to category name
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    names = []
    for c in classes:
        names.append(cat_to_name[c])
    
    return names


def main():
    ''' Create & retrieve Command Line arguments. '''
    in_args = get_input_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() and in_args.gpu else 'cpu')
    model, class_idx = load_checkpoint(in_args.checkpoint, in_args.gpu)
    probs, classes = predict(in_args.input, model, class_idx, in_args.top_k, in_args.gpu)
    names = predict_names(classes, in_args.category_names)

    print(dict(zip(names, probs)))

if __name__ == '__main__':
	main()
