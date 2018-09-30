from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt

from skimage import io, transform
from torchvision import transforms, utils

import numpy as np


class FvDataset(Dataset):
    def __init__(self, csv_in_dir, csv_out, n_feature, transform=None):
        """
            Args:
                csv_file (string): Path to the csv file with force torque values and position of pressing.
                init_img_path (string): path to the  init image
                imgs_path (string): path to the image when pressing
                transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_in_dir = csv_in_dir
        self.csv_out_handler = pd.read_csv(csv_out)
        self.n_feature = n_feature
        self.transform = transform

    def __len__(self):
        return len(self.csv_out_handler)

    def __getitem__(self, idx):
        csv_in_name = '%04d.csv' % idx
        feature_in_path = os.path.join(self.csv_in_dir, csv_in_name)
        # print(feature_in_path)

        feature_in = pd.read_csv(feature_in_path).values

        # feature_out = self.csv_out_handler.iloc[idx, 0:self.n_feature].values
        feature_out = self.csv_out_handler.iloc[idx, 2:self.n_feature].values
        # print feature_out
        # targets = targets.astype('float').reshape(-1, 2)
        sample = {'feature_in': feature_in, 'feature_out': feature_out}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        feature_in, feature_out = sample['feature_in'], sample['feature_out']
        feature_in = feature_in.astype(float)
        feature_out = feature_out.astype(float)  # origin format too long to use from_numpy

        tensor_in = torch.Tensor(feature_in)
        tensor_out = torch.Tensor(feature_out)
        # print tensor_out.shape
        # flatten tensor_in
        n_in = tensor_in.shape

        tensor_in = tensor_in.view(n_in[0]*n_in[1], -1)
        tensor_in = tensor_in.squeeze(1)

        return {'feature_in': tensor_in,
                'feature_out': tensor_out}




if __name__ == '__main__':
    fv_dataset = FvDataset(csv_in_dir='../dataset2/interpolation/vectors',
                           csv_out='../dataset2/wrench_loc_normalized.csv',
                           n_feature=6,
                           transform=transforms.Compose([
                               ToTensor()])
                           )
    print(len(fv_dataset))

    sample = fv_dataset[100]
    print sample['feature_out'].shape

    print os.getcwd()
    # fig = plt.figure()

    # for i in range(len(fv_dataset)):
    #     sample = fv_dataset[i]
    #
    #     # print(i, sample['image'].size(), sample['init_image'].size(), sample['targets'].size())
    #     print(i, sample['feature_in'].shape, sample['feature_out'])




