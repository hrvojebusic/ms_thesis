import argparse
import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

import models
import utils.data_loader as data_loader

# Configuration
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='training dataset: mnist | cifar10')
parser.add_argument('--data_normalization', action='store_true', help='normalize datasets')
parser.add_argument('--data_root', required=True, help='path to dataset')
parser.add_argument('--out_folder', required=True, help='folder to output images and model checkpoints to')
parser.add_argument('--chp_freq', type=int, default=10, help='model checkpoint frequency (in epochs)')
parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--kl_beta', type=float, default=1., help='KL divergence hyperparameter')
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--classifier', help='additional classifier to train generator with')
parser.add_argument('--cuda', default=True, action='store_true')
parser.add_argument('--gpu_index', type=int, default=1, help='GPU used for training, defaults to: "1"')
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

discriminator = models.get_discriminator(channels, args.ndf)
generator = models.get_generator(args.nz, channels, args.ngf)
if gpu_index:
    discriminator = discriminator.cuda(gpu_index)
    generator = generator.cuda(gpu_index)

if args.classifier:
    # classifier = models.get_classifier(channels, classes)
    classifier = models.vgg13()
    classifier.load_state_dict(torch.load(args.classifier))
    if gpu_index:
        classifier = classifier.cuda(gpu_index)
    classifier = classifier.eval()
    print('Using pre-trained classifier. Classifier loaded from:\n%s' % args.classifier)

fixed_noise = torch.randn(args.batch_size, args.nz, 1, 1, device=device)

criterion = nn.BCELoss()
d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))

for epoch in range(1, args.epochs + 1):
    for i, data in enumerate(train_data):
        images, _ = data
        true_batch_size = images.size(0)
        if gpu_index:
            images = images.cuda(gpu_index)

        real_label = torch.full((true_batch_size,), 1, device=device)
        fake_label = torch.full((true_batch_size,), 0, device=device)

        ########################
        # (1) Update D network #
        ########################
        discriminator.zero_grad()

        # Train with real
        output = discriminator(images)
        err_d_real = criterion(output, real_label)
        err_d_real.backward()
        D_x = output.mean().item()

        # Train with fake
        noise = torch.randn(true_batch_size, args.nz, 1, 1, device=device)
        fake_images = generator(noise)
        output = discriminator(fake_images.detach())
        err_d_fake = criterion(output, fake_label)
        err_d_fake.backward()
        D_G_z1 = output.mean().item()

        err_d = err_d_real + err_d_fake
        d_optimizer.step()

        ########################
        # (2) Update G network #
        ########################
        generator.zero_grad()

        output = discriminator(fake_images)
        err_g = criterion(output, real_label)
        D_G_z2 = output.mean().item()

        if args.classifier:
            output = F.log_softmax(classifier(fake_images), dim=1)
            uniform_dist = torch.zeros((true_batch_size, 10), device=device).fill_((1. / 10))
            err_kl = args.kl_beta * F.kl_div(output, uniform_dist, reduction='batchmean')
            err_g += err_kl

        err_g.backward()
        g_optimizer.step()

        if i % 100 == 99:
            if args.classifier:
                print(
                    '[%3d|%3d] Loss_D: %.4f Loss_G: %.4f Loss_C: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                    % (epoch, i + 1, err_d.item(), err_g.item(), err_kl.item(), D_x, D_G_z1, D_G_z2)
                )
            else:
                print(
                    '[%3d|%3d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                    % (epoch, i + 1, err_d.item(), err_g.item(), D_x, D_G_z1, D_G_z2)
                )

    fake = generator(fixed_noise)
    fake_save_path = os.path.join(args.out_folder, 'fake_samples_epoch_%d.png' % epoch)
    vutils.save_image(fake.detach(), fake_save_path, normalize=True)

    if epoch % args.chp_freq == 0:
        discriminator_save_path = 'discriminator_%s_epoch_%d.pth' % (args.dataset, epoch)
        discriminator_save_path = os.path.join(args.out_folder, discriminator_save_path)

        generator_save_path = 'generator_%s_epoch_%d.pth' % (args.dataset, epoch)
        generator_save_path = os.path.join(args.out_folder, generator_save_path)

        torch.save(discriminator.state_dict(), discriminator_save_path)
        torch.save(generator.state_dict(), generator_save_path)
