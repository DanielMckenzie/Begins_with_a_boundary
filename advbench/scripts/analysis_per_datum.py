import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse







def plot_distribution(args):
    pickle_file_path = os.path.join('advbench/train-output', args.model_name, 'Per_Datum_results.pkl')

    # Load the data
    with open(pickle_file_path, 'rb') as file:
        results = pickle.load(file)

    data = results[args.split]

    # Separate correctly and incorrectly classified data
    # correct = data[data[:, 1] == 1]

    incorrect = data[data[:, 1] == 0]
    num_incorrect = incorrect.shape[0]
    print(num_incorrect)
    save_fig_name = os.path.join('advbench/figs/', args.model_name+'_'+args.split+'.pdf')
    pathological_images = incorrect[incorrect[:,0] > 94]
    num_pathological_images = pathological_images.shape[0]
    print(num_pathological_images)

    # Plotting
    plt.figure(figsize=(12, 6))
    # plt.hist(correct[:, 0], bins=50, alpha=0.5, label='Correctly Classified')
    plt.hist(incorrect[:, 0], bins=50, alpha=0.5, label='Incorrectly Classified')
    plt.xlabel('Percentage of perturbations yielding correct classification')
    plt.ylabel('Number of Images')
    plt.axvline(x=95, color='red')
    # plt.title(title)
    plt.legend()
    plt.savefig(save_fig_name)

# Plot for Train, Validation, and Test
# plot_distribution(results['train'], 'Train Data Distribution')
# plot_distribution(results['val'], 'Validation Data Distribution')
# plot_distribution(results['test'], 'Test Data Distribution')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collate and plot data")
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--model_name', type=str, default='cifar_005')
    args = parser.parse_args()
    plot_distribution(args)


