import argparse
import torch
from torch.utils.data import DataLoader
import os
import json
import pandas as pd
import pickle
import time
import collections
from humanfriendly import format_timespan

from advbench import datasets
from advbench import algorithms
from advbench import evalulation_methods
from advbench import hparams_registry
from advbench.lib import misc, meters, reporting

def main(args, hparams, test_hparams):

    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'mps' # for running on apple silicon
    
    # paths for saving output
    json_path = os.path.join(args.output_dir, 'results.json')
    ckpt_path = misc.stage_path(args.output_dir, 'ckpts')
    train_df_path = os.path.join(args.output_dir, 'train.pd')
    selection_df_path = os.path.join(args.output_dir, 'selection.pd')
    device = args.device
    torch.manual_seed(0)

    dataset = vars(datasets)[args.dataset](args.data_dir, device)

    train_loader = DataLoader(
        dataset=dataset.splits['train'],
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS,
        pin_memory=False,
        shuffle=True)
    validation_loader = DataLoader(
        dataset=dataset.splits['val'],
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS,
        pin_memory=False,
        shuffle=False)
    test_loader = DataLoader(
        dataset=dataset.splits['test'],
        batch_size=100,
        num_workers=dataset.N_WORKERS,
        pin_memory=False,
        shuffle=False)

    algorithm = vars(algorithms)[args.algorithm](
        dataset.INPUT_SHAPE, 
        dataset.NUM_CLASSES,
        hparams,
        device).to(device)

    def save_checkpoint(epoch):
        torch.save(
            obj={'state_dict': algorithm.state_dict()}, 
            f=os.path.join(ckpt_path, f'model_ckpt_{epoch}.pkl')
        )

    def load_checkpoint(epoch):
        ckpt = torch.load(os.path.join(ckpt_path, f'model_ckpt_{epoch}.pkl'), map_location=torch.device('mps'))
        algorithm.load_state_dict(ckpt['state_dict'])



    load_checkpoint('final')

    evaluator = vars(evalulation_methods)['Per_Datum'](
            algorithm=algorithm,
            device=device,
            # output_dir=args.output_dir,
            test_hparams=test_hparams)

    val_res = evaluator.calculate(validation_loader)
    print('val_res: ', val_res)
    test_res = evaluator.calculate(test_loader)
    print('test_res: ', test_res)
    train_res = evaluator.calculate(train_loader)
    print('train_res: ', train_res)

    # save results as numpy arrays stored in a dictionary
    results = {
        'train': train_res.cpu().numpy(),
        'val': val_res.cpu().numpy(),
        'test': test_res.cpu().numpy(),
    }

    # save results as a pickle file
    with open(os.path.join(args.output_dir, 'Per_Datum_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Adversarial robustness')
    parser.add_argument('--data_dir', type=str, default='./advbench/data')
    parser.add_argument('--output_dir', type=str, default='train_output')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset to use')
    parser.add_argument('--algorithm', type=str, default='ERM', help='Algorithm to run')
    parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0, help='Seed for hyperparameters')
    parser.add_argument('--trial_seed', type=int, default=0, help='Trial number')
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--evaluators', type=str, nargs='+', default=['Clean'])
    parser.add_argument('--save_model_every_epoch', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0', help='Select device')
    parser.add_argument('--cvar_sgd_beta', type=float, default=None, help='CVaR-SGD beta')
    parser.add_argument('--epsilon', type=float, default=None, help='Epsilon for PGD attack')
    parser.add_argument('--cvar_sgd_t_step_size', type=float, default=None, help='CVaR-SGD t step size')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir), exist_ok=True)

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.dataset not in vars(datasets):
        raise NotImplementedError(f'Dataset {args.dataset} is not implemented.')

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        seed = misc.seed_hash(args.hparams_seed, args.trial_seed)
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset, seed)
        
    if args.cvar_sgd_beta is not None:
        hparams.update({'cvar_sgd_beta': args.cvar_sgd_beta})

    if args.cvar_sgd_t_step_size is not None:
        hparams.update({'cvar_sgd_t_step_size': args.cvar_sgd_t_step_size})

    print ('Hparams:')
    for k, v in sorted(hparams.items()):
        print(f'\t{k}: {v}')

    with open(os.path.join(args.output_dir, 'hparams.json'), 'w') as f:
        json.dump(hparams, f, indent=2)

    test_hparams = hparams_registry.test_hparams(args.algorithm, args.dataset)

    if args.epsilon is not None:
        test_hparams.update({'epsilon': args.epsilon})

    print('Test hparams:')
    for k, v in sorted(test_hparams.items()):
        print(f'\t{k}: {v}')

    with open(os.path.join(args.output_dir, 'test_hparams.json'), 'w') as f:
        json.dump(test_hparams, f, indent=2)

    main(args, hparams, test_hparams)