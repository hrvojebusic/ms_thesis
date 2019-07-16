import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import models
import utils.data_loader as data_loader


def plot_predictive_probability_distribution(probabilities, color, label, fig_path, step=0.05, plot_ideal=True):
    bins = np.arange(0, 1.1, step=step)
    
    hist, bins = np.histogram(probabilities, bins)
    hist = hist / (1.0 * len(probabilities))
    
    p = plt.bar(bins[:-1], hist, 0.8 * step, align='center', color=color, edgecolor='black')
    
    plt.xticks(np.arange(0, 1.1, step=0.1))
    plt.xlabel('Najveći iznos prediktivne distribucije')
    
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.ylabel('Udio')
    
    if plot_ideal:
        plt.axvline(0.1, color='red', linestyle='dashed')
    
    plt.legend([p], [label])
    
    plt.savefig(fig_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True, help='path to datasets')
    parser.add_argument('--data_normalization', action='store_true', help='normalize datasets')
    parser.add_argument('--in_dataset', required=True, help='dataset used for training: svhn | cifar10')
    parser.add_argument('--out_datasets', required=True, nargs='+', help='datasets used for testing')
    parser.add_argument('--model_chp', required=True, help='path to VGG13 model checkpoint')
    parser.add_argument('--out_folder', required=True, help='folder to output images to')
    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--gpu_index', type=int, default=0, help='GPU used for training, defaults to: "0"')
    args = parser.parse_args()

    batch_size = 128
    mappings = {
        'svhn': 'SVHN',
        'cifar10': 'CIFAR-10',
        'imagenet': 'Imagenet',
        'lsun': 'LSUN'
    }

    if args.cuda:
        gpu_index = args.gpu_index
    else:
        gpu_index = None

    _, in_test_data, _, _ = data_loader.get_train_data(
        args.in_dataset, args.data_root, batch_size, normalize=args.data_normalization
    )

    out_test_data = []
    for dataset in args.out_datasets:
        test_data, _, _ = data_loader.get_test_data(
                dataset, args.data_root, batch_size, normalize=args.data_normalization
            )
        out_test_data.append((dataset, test_data))

    model = models.vgg13()
    model.load_state_dict(torch.load(args.model_chp))
    if args.cuda:
        model = model.cuda(gpu_index)
    model.eval()

    # In-dataset
    scores_in = np.array([])

    for data, _ in in_test_data:
        if args.cuda:
            data = data.cuda(gpu_index)

        logits = model(data)
        probs = F.softmax(logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
        max_probs = max_probs.detach().cpu()
        scores_in = np.concatenate((scores_in, max_probs), axis=0)

    # Plot in-dataset
    pd_in_path = os.path.join(args.out_folder, 'pd_in_%s.png' % args.in_dataset)
    plot_predictive_probability_distribution(
        scores_in, 'red', mappings[args.in_dataset], pd_in_path, plot_ideal=False
    )
    print('Plot for %s saved to: %s' % (args.in_dataset, pd_in_path))

    for dataset, test_loader in out_test_data:
        scores_out = np.array([])

        for data, _ in test_loader:
            if args.cuda:
                data = data.cuda(gpu_index)

            logits = model(data)
            probs = F.softmax(logits, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            max_probs = max_probs.detach().cpu()
            scores_out = np.concatenate((scores_out, max_probs), axis=0)
            
        # Plot out-dataset
        pd_out_path = os.path.join(args.out_folder, 'pd_out_%s.png' % dataset)
        plot_predictive_probability_distribution(
            scores_out, 'blue', mappings[dataset], pd_out_path
        )
        print('Plot for %s saved to: %s' % (dataset, pd_out_path))


if __name__=='__main__':
    main()
