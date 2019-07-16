import os

import torchvision.utils as vutils

import utils.data_loader as data_loader


def image_sampler(data_root, save_dir):
    # Train data
    mnist, _, _, _ = data_loader.get_train_data('mnist', data_root, 64)
    svhn, _, _, _ = data_loader.get_train_data('svhn', data_root, 64)
    cifar10, _, _, _ = data_loader.get_train_data('cifar10', data_root, 64)

    # Test data
    fsh_mnist, _, _ = data_loader.get_test_data('fashion-mnist', data_root, 64)
    imagenet, _, _ = data_loader.get_test_data('imagenet', data_root, 64)
    lsun, _, _ = data_loader.get_test_data('lsun', data_root, 64)

    vutils.save_image(
        mnist.__iter__().__next__()[0],
        os.path.join(save_dir, 'mnist_sample.png'),
        normalize=True
    )
    vutils.save_image(
        svhn.__iter__().__next__()[0],
        os.path.join(save_dir, 'svhn_sample.png'),
        normalize=True
    )
    vutils.save_image(
        cifar10.__iter__().__next__()[0],
        os.path.join(save_dir, 'cifar10_sample.png'),
        normalize=True
    )
    vutils.save_image(
        fsh_mnist.__iter__().__next__()[0],
        os.path.join(save_dir, 'fsh-mnist_sample.png'),
        normalize=True
    )
    vutils.save_image(
        imagenet.__iter__().__next__()[0],
        os.path.join(save_dir, 'imagenet_sample.png'),
        normalize=True
    )
    vutils.save_image(
        lsun.__iter__().__next__()[0],
        os.path.join(save_dir, 'lsun_sample.png'),
        normalize=True
    )

    print('Samples generated and saved.')


if __name__ == '__main__':
    data_root = os.path.join('..', 'data')
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    save_dir = os.path.join('..', 'results', 'image_samples')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_sampler(data_root, save_dir)
