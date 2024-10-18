import numpy as np
from torchvision.datasets import MNIST
import json

def download_mnist(is_train: bool):
    dataset = MNIST(root='./data', transform=lambda x: np.array(x).flatten(), download=True, train=is_train)
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    return mnist_data, mnist_labels

def test_data(data, label, weights, biases):
    weighted_sums = data.dot(weights) + biases
    #exponential sum
    exp_weighted_sums = np.exp(weighted_sums - np.max(weighted_sums, axis=0, keepdims=True))
    probabilities = exp_weighted_sums / np.sum(exp_weighted_sums, axis=0, keepdims=True)
    answer = np.argmax(probabilities)
    return answer == label

def calculate_accuracy(epochs, data, labels) -> int:
    #get calculated data from json
    json_name = f'calculated_data_{epochs}.json'
    with open(json_name, 'r') as json_file:
        calculated_data = json.load(json_file)
    weights_list = calculated_data['weights']
    biases_list = calculated_data['biases']
    weights = np.array(weights_list)
    biases = np.array(biases_list)

    correct_tests = 0
    for index in range(len(data)):
        if test_data(np.array(data[index]), labels[index], weights, biases):
            correct_tests += 1

    return float(correct_tests / 100) #percentage 100% is 10000

test_X, test_Y = download_mnist(False)
print(f"Accuracy for training 50 epochs: {calculate_accuracy(50, test_X, test_Y)}")
print(f"Accuracy for training 500 epochs: {calculate_accuracy(500, test_X, test_Y)}")


