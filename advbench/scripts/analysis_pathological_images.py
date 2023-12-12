import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch

def plot_images(args):
    pickle_file_path = os.path.join('advbench/train-output', args.model_name, 'Per_Datum_results.pkl')

    # Load the data
    with open(pickle_file_path, 'rb') as file:
        data = pickle.load(file)

    path_images = torch.cat(data['path_images'], dim=0)
    perturbation_preds = torch.cat(data['perturbation_preds'], dim=0)
    perturbed_imgs = torch.cat(data['perturbed_imgs'], dim=0)
    print(perturbation_preds[2,:])

    for i in range(3):
        save_fig_name = os.path.join('advbench/figs/', args.model_name+'_'+'test_'+'pathological_image_'+str(i)+'.pdf')
        base_image_numpy = path_images[i,:,:,:].squeeze().permute(1, 2, 0).cpu().numpy()
        plt.imshow(base_image_numpy)
        plt.axis('off')
        plt.savefig(save_fig_name)
        num_pert_imgs = 0
        j = 0
        while num_pert_imgs <= 3:
            if perturbation_preds[i,j] == 1:
                save_fig_name = os.path.join('advbench/figs/', args.model_name+'_'+str(i)+'_perturbed_image_'+ str(num_pert_imgs) + '.pdf')
                pert_image_numpy = perturbed_imgs[i,j,:,:,:].squeeze().permute(1, 2, 0).cpu().numpy()
                plt.imshow(pert_image_numpy)
                plt.axis('off')
                plt.savefig(save_fig_name)
                num_pert_imgs += 1
                print(num_pert_imgs)
            j+=1
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and plot pathological images")
    parser.add_argument('--split', type=str, default="train")
    ## TODO
    # See how split is handled.
    parser.add_argument('--model_name', type=str, default='cifar_005')
    args = parser.parse_args()
    plot_images(args)


