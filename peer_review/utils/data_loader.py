import os

import torch.utils.data as data_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms

from utils.fashion_mnist import FashionMNIST


def _svhn_target_transform(target):
    new_target = target - 1
    if new_target == -1:
        new_target = 9
    return new_target


def get_train_data(dataset, data_root, batch_size, image_size=32, normalize=True):
    if normalize:
        if dataset == 'mnist':
            normalization = transforms.Normalize((0.5,), (0.5,))
        else:
            normalization = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalization
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    if dataset == 'mnist':
        data_root = os.path.join(data_root, 'mnist')
        train_set = dset.MNIST(root=data_root, train=True, download=True, transform=transform)
        test_set = dset.MNIST(root=data_root, train=False, download=True, transform=transform)
        channels = 1
        classes = 10

    elif dataset == 'svhn':
        data_root = os.path.join(data_root, 'svhn')
        train_set = dset.SVHN(
            root=data_root, split='train', download=True,
            transform=transform, target_transform=_svhn_target_transform
        )
        test_set = dset.SVHN(
            root=data_root, split='test', download=True,
            transform=transform, target_transform=_svhn_target_transform
        )
        channels = 3
        classes = 10

    elif dataset == 'cifar10':
        data_root = os.path.join(data_root, 'cifar10')
        train_set = dset.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        test_set = dset.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        channels = 3
        classes = 10

    else:
        raise ValueError('Unrecognized dataset: %s' % dataset)

    train_loader = data_utils.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = data_utils.DataLoader(test_set,  batch_size=batch_size, shuffle=True, num_workers=1)

    return train_loader, test_loader, channels, classes


def get_test_data(dataset, data_root, batch_size, image_size=32, normalize=True):
    if normalize:
        if dataset == 'fashion-mnist':
            normalization = transforms.Normalize((0.5,), (0.5,))
        else:
            normalization = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalization
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    if dataset == 'fashion-mnist':
        data_root = os.path.join(data_root, 'fashion-mnist')
        test_set = FashionMNIST(root=data_root, train=False, download=True, transform=transform)
        channels = 1
        classes = 10

    elif dataset == 'svhn':
        data_root = os.path.join(data_root, 'svhn')
        test_set = dset.SVHN(
            root=data_root, split='test', download=True,
            transform=transform, target_transform=_svhn_target_transform
        )
        channels = 3
        classes = 10

    elif dataset == 'cifar10':
        data_root = os.path.join(data_root, 'cifar10')
        test_set = dset.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        channels = 3
        classes = 10

    elif dataset == 'imagenet':
        data_root = os.path.join(data_root, 'Imagenet_resize')
        test_set = dset.ImageFolder(root=data_root, transform=transform)
        channels = 3
        classes = 10

    elif dataset == 'lsun':
        data_root = os.path.join(data_root, 'LSUN_resize')
        test_set = dset.ImageFolder(root=data_root, transform=transform)
        channels = 3
        classes = 10

    else:
        raise ValueError('Unrecognized dataset: %s' % dataset)

    test_loader = data_utils.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)

    return test_loader, channels, classes
