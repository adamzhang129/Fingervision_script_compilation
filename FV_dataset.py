from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt

from skimage import io, transform
from torchvision import transforms, utils

import numpy as np


class FvDataset(Dataset):
    def __init__(self, csv_file, n_feature,  init_img_path, imgs_dir, transform=None):
        """
            Args:
                csv_file (string): Path to the csv file with force torque values and position of pressing.
                init_img_path (string): path to the  init image
                imgs_path (string): path to the image when pressing
                transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_handler = pd.read_csv(csv_file)
        self.n_feature = n_feature
        self.init_img_path = init_img_path
        self.imgs_dir = imgs_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_handler)

    def __getitem__(self, idx):
        imgs_name = os.path.join(self.imgs_dir, self.csv_handler.iloc[idx, 0])
        init_img_name = self.init_img_path

        imgs = io.imread(imgs_name, plugin='matplotlib')
        init_img = io.imread(init_img_name, plugin='matplotlib')

        targets = self.csv_handler.iloc[idx, 1:self.n_feature+1].as_matrix()
        # print(targets)
        # targets = targets.astype('float').reshape(-1, 2)
        sample = {'image': imgs, 'init_image': init_img, 'targets': targets}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, init_image, targets = sample['image'], sample['init_image'], sample['targets']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        img_init = transform.resize(init_image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        # landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'init_image': img_init, 'targets': targets}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, init_image, targets = sample['image'], sample['init_image'], sample['targets']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # add channel dimension to accommodate torch image fashion
        image = image.reshape((-1, image.shape[0], image.shape[1]))
        init_image = init_image.reshape((-1, init_image.shape[0], init_image.shape[1]))
        targets = targets.astype(float)  # origin format too long to use from_numpy

        # image = image.transpose((2, 0, 1))
        # init_image = init_image.transpose((2,0,1))
        return {'image': torch.from_numpy(image),
                'init_image': torch.from_numpy(init_image),
                'targets': torch.from_numpy(targets)}




if __name__ == '__main__':
    fv_dataset = FvDataset(csv_file='../loc_fxyz_mz_normalized.csv',
                           init_img_path='../fingercam_undistorted_equalsize/init/img_init.jpg',
                           imgs_dir='../fingercam_undistorted_equalsize/displacement',
                           transform=transforms.Compose([
                               Rescale(256),
                               ToTensor()])
                           )
    print(len(fv_dataset))

    sample = fv_dataset[100]

    fig = plt.figure()

    for i in range(len(fv_dataset)):
        sample = fv_dataset[i]

        # print(i, sample['image'].size(), sample['init_image'].size(), sample['targets'].size())
        print(i, sample['image'].size(), sample['init_image'], sample['targets'])

        ax = plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        ax.set_title('sample #{}'.format(i))
        ax.set_axis_off()

        plt.imshow(sample['image'].reshape((256, 256)), cmap='gray')

        if i == 5:
            plt.show()
            break


# for i in range(len(fv_dataset)):
#     sample = fv_dataset[i]
#
#     # print(i, sample['target_vectors'])
#     print(transform.resize( sample['image'], (256, 256)).shape)
#     ax = plt.subplot(2, 3, i+1)
#     plt.tight_layout()
#     ax.set_title('sample #{}'.format(i))
#     ax.set_axis_off()
#
#     plt.imshow(sample['init_image'], cmap='gray')
#
#     if i == 5:
#         plt.show()
#         break
