import pickle
import os
import matplotlib.pyplot as plt
import numpy as np


pickle_file_path = 'advbench/train-output/cifar_005/Per_Datum_results.pkl'

# Load the data
with open(pickle_file_path, 'rb') as file:
    results = pickle.load(file)


def plot_distribution(data, title):
    # Separate correctly and incorrectly classified data
    # correct = data[data[:, 1] == 1]
    incorrect = data[data[:, 1] == 0]

    # Plotting
    plt.figure(figsize=(12, 6))
    # plt.hist(correct[:, 0], bins=50, alpha=0.5, label='Correctly Classified')
    plt.hist(incorrect[:, 0], bins=50, alpha=0.5, label='Incorrectly Classified')
    plt.xlabel('Percentage Accuracy')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.show()

# Plot for Train, Validation, and Test
plot_distribution(results['train'], 'Train Data Distribution')
plot_distribution(results['val'], 'Validation Data Distribution')
plot_distribution(results['test'], 'Test Data Distribution')