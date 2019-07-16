import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim as optim

import models
import utils.data_loader as data_loader

# Configuration
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='training dataset: mnist | svhn | cifar10')
parser.add_argument('--data_normalization', action='store_true', help='normalize datasets')
parser.add_argument('--data_root', required=True, help='path to dataset')
parser.add_argument('--out_folder', required=True, help='folder to output results and model checkpoints to')
parser.add_argument('--chp_freq', type=int, default=10, help='model checkpoint frequency (in epochs)')
parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--kl_beta', type=float, default=1., help='KL divergence hyperparameter')
parser.add_argument('--generator', help='generator for marginal samples')
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpu_index', type=int, default=0, help='GPU used for training, defaults to: "0"')
args = parser.parse_args()

# Check
print(args)

image_size = 32  # Immutable

if args.cuda:
    gpu_index = args.gpu_index
    torch.cuda.set_device(gpu_index)
    device = torch.device('cuda')
else:
    gpu_index = None
    device = torch.device('cpu')

train_data, _, channels, classes = data_loader.get_train_data(
    args.dataset, args.data_root, args.batch_size, normalize=args.data_normalization)

# classifier = models.get_classifier(channels, classes)
classifier = models.vgg13()
if gpu_index:
    classifier = classifier.cuda(gpu_index)

if args.generator:
    generator = models.get_generator(args.nz, channels, args.ngf)
    generator.load_state_dict(torch.load(args.generator))
    if gpu_index:
        generator = generator.cuda(gpu_index)
    generator = generator.eval()
    print('Using pre-trained generator. Generator loaded from:\n%s' % args.generator)

optimizer = optim.Adam(classifier.parameters(), lr=args.lr, betas=(0.5, 0.999))

run_loss = run_real = run_fake = 0.

for epoch in range(1, args.epochs + 1):
    for i, data in enumerate(train_data):
        optimizer.zero_grad()

        # Train with real
        images, labels = data
        if gpu_index:
            images = images.cuda(gpu_index)
            labels = labels.cuda(gpu_index)

        true_batch_size = images.size(0)
        uniform_dist = torch.zeros(true_batch_size, 10).fill_(1. / 10)
        if gpu_index:
            uniform_dist = uniform_dist.cuda(gpu_index)

        outputs_real = F.log_softmax(classifier(images), dim=1)
        loss_real = F.nll_loss(outputs_real, labels)
        loss_real.backward()

        # Train with fake
        if args.generator:
            noise = torch.randn(true_batch_size, args.nz, 1, 1)
            if gpu_index:
                noise = noise.cuda(gpu_index)
            noise = generator(noise)
        else:
            noise = torch.randn(true_batch_size, channels, image_size, image_size)
            if gpu_index:
                noise = noise.cuda(gpu_index)

        outputs_fake = F.log_softmax(classifier(noise), dim=1)
        loss_fake = args.kl_beta * F.kl_div(outputs_fake, uniform_dist, reduction='batchmean')
        loss_fake.backward()

        loss = loss_real + loss_fake
        optimizer.step()

        # Statistics
        run_loss += loss.item()
        run_real += loss_real.item()
        run_fake += loss_fake.item()

        if i % 100 == 99:
            print('[%3d|%3d] Loss: %.6f | Real: %.6f | Fake: %.6f' %
                  (epoch, i + 1, run_loss / 100, run_real / 100, run_fake / 100))
            run_loss = run_real = run_fake = 0.

    print('Finished epoch [%3d|%3d]' % (epoch, args.epochs))
    if epoch % args.chp_freq == 0:
        save_path = os.path.join(args.out_folder, 'classifier_%s_epoch_%d.pth' % (args.dataset, epoch))
        torch.save(classifier.state_dict(), save_path)
