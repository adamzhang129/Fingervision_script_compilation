import torch
import torch.nn as nn
import torch.optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import Adam, SGD
from torch.autograd import Variable
import numpy as np

from torch.nn import init

import torchvision.models as models

import matplotlib.pyplot as plt
from FV_dataset import *
import os

EPOCH = 5
BATCH_SIZE = 4
LR = 0.00001
momentum = 0.99
n_feature = 6

# load fv dataset
csv_file = '../loc_fxyz_mz.csv'
displacement_image_path = '../fingercam_undistorted/displacement'
init_image_path = '../fingercam_undistorted/init/img_init.jpg'
if not (os.path.isfile(csv_file)):
    print('Dataset csv file does not exist')
if not (os.path.exists(displacement_image_path)) or not (os.listdir(displacement_image_path)):
    print('displacement path not exist or dir is empty')
if not (os.path.isfile(init_image_path)):
    print('init image not exist')

fv_dataset = FvDataset(csv_file='../loc_fxyz_mz_normalized.csv',
                       n_feature = n_feature,
                       init_img_path='../fingercam_undistorted_equalsize/init/img_init.jpg',
                       imgs_dir='../fingercam_undistorted_equalsize/displacement',
                       transform=transforms.Compose([
                           Rescale(256),
                           ToTensor()]),
                       )
# sample = fv_dataset[10]
# print(sample['targets'])

# fig = plt.figure()
#
# for i in range(len(fv_dataset)):
#     sample = fv_dataset[i]
#
#     print(i, sample['image'].size(), sample['init_image'].size(), sample['targets'].size())
#
#     ax = plt.subplot(2, 3, i + 1)
#     plt.tight_layout()
#     ax.set_title('sample #{}'.format(i))
#     ax.set_axis_off()
#
#     plt.imshow(sample['image'].reshape((256, 256)), cmap='gray')
#
#     if i == 5:
#         plt.show()
#         break

train_loader = DataLoader(dataset=fv_dataset, batch_size=BATCH_SIZE, shuffle=True)


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        print('conv weight initializing')
        m.weight.data.normal_(0.0, 0.02)
    if classname.find('BatchNorm') !=-1:
        print('bn weight initializing')
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class FvNet(nn.Module):
    def __init__(self, n_input=256, feature_size=6):
        super(FvNet, self).__init__()

        self.feature1 = nn.Sequential(  # input shape (1, n, n)
            nn.Conv2d(
                in_channels=1,  # input height == color channels of input image
                out_channels=16,  # n_filter
                kernel_size=5,  # filter nXn
                stride=1,  # filter slide step size
                padding=2,
                # to preserve size of input to output, use padding = (kernel_size -1)/2, leave downsampling to pooling
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # %75 downsampling --> output shape (16, n/4, n/4)

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output shape (32, n/8, n/8)

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output shape (64, n/16, n/16)

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output shape (64, n/32, n/32)

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output shape (64, n/64, n/64)

            nn.Dropout2d(0.7)
        )

        self.feature2 = nn.Sequential(  # input shape (1, n, n)
            nn.Conv2d(
                in_channels=1,  # input height == color channels of input image
                out_channels=16,  # n_filter
                kernel_size=5,  # filter nXn
                stride=1,  # filter slide step size
                padding=2,
                # to preserve size of input to output, use padding = (kernel_size -1)/2, leave downsampling to pooling
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # %75 downsampling --> output shape (16, n/4, n/4)

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output shape (32, n/8, n/8)

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output shape (64, n/16, n/16)

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output shape (64, n/32, n/32)

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output shape (64, n/64, n/64)

            nn.Dropout2d(0.7)

        )

        self.regressor = nn.Sequential(
            nn.Linear(2 * 64 * int(n_input / 32) * int(n_input / 32), 1024),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(128, feature_size)
        )

    # def _set_init_(self, layer): # params initialization
    #     init.normal_(layer.weight, mean=0., std= 0.1)


    def forward(self, img_init, imgs):
        f1 = self.feature1(img_init)
        f2 = self.feature2(imgs)
        f_cat = torch.cat((f1, f2), 1)

        f_cat = f_cat.view(f_cat.size(0), -1)
        # print(f_cat.size())
        f_cat = self.regressor(f_cat)
        return f_cat


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('CUDA available?: ' + str(torch.cuda.is_available()))
net = FvNet(feature_size=n_feature).cuda()
print(net)

net.apply(weight_init)


optimizer = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.5, 0.999))
# optimizer = SGD(net.parameters(), lr=LR, momentum=momentum)
loss_fn = nn.MSELoss(reduce=True, size_average=True)

from matplotlib import cm

try:
    from sklearn.manifold import TSNE;

    HAS_SK = True
except:
    HAS_SK = False;
    print('install sklearn for layer visualization')


def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9));
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max());
    plt.ylim(Y.min(), Y.max());
    plt.title('Visualize last layer');
    plt.show();
    plt.pause(0.01)


def plot_kernels(tensor, num_cols=6):
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    # if not tensor.shape[-1]==3:
    #     raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)

        # tensor[i] = tensor[i].reshape((5,5))
        # print(tensor[i].shape)
        ax1.imshow(tensor[i].reshape((tensor[i].shape[-1], tensor[i].shape[-1])), cmap='gray')
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # plt.show()

plt.ion()
loss_plot = np.zeros((EPOCH * 187, 1))

import visdom
viz = visdom.Visdom()
win = []
layer = []

for epoch in range(EPOCH):
    print('new epoch {}'.format(epoch))
    for step, sample_batched in enumerate(train_loader):
        # print(sample_batched['image'].size())
        img = sample_batched['image'].float().cuda()
        # need to add .float() to fix bug: https://github.com/pytorch/pytorch/issues/2138
        img_init = sample_batched['init_image'].float().cuda()
        target = sample_batched['targets'].float().cuda()
        print(target)
        # print('target vector:'+ str(target))
        # print(img.type(), img_init.type(), target.type())
        output = net(img_init, img)

        loss = loss_fn(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if step % 50 == 0:
        print('epoch: ', epoch, 'step:', step, 'loss: ', loss.cpu().data.numpy())
        loss_plot[step + epoch * 187] = loss.cpu().data.numpy()
        # print(np.linspace(0, step + epoch*187,step + epoch*187).shape,loss_plot[:(step + epoch*187)].shape )
        if (step + epoch * 187)> 0:
            plt.cla()
            # plt.plot(np.linspace(0, step + epoch * 187, step + epoch * 187).reshape(-1,1), loss_plot[:(step + epoch * 187)])
            # plt.plot(step + epoch * 187, loss.cpu().data.numpy(), marker='*')
            # plt.pause(0.1)
            # plt.draw()

            layer1 = [i for i in net.children()][0]
            layer1 = layer1[0].weight.cpu().data
            # plot_kernels(layer1)
            layer = viz.images(layer1, win =layer, opts=dict(colormap='Greys'))
            # win = viz.matplot(plt, win=win)

for step, sample_batched in enumerate(train_loader):
    # print(sample_batched['image'].size())
    img = sample_batched['image'].float().cuda()
    # need to add .float() to fix bug: https://github.com/pytorch/pytorch/issues/2138
    img_init = sample_batched['init_image'].float().cuda()
    target = sample_batched['targets'].float().cuda()
    # print('target vector:'+ str(target))
    # print(img.type(), img_init.type(), target.type())
    output = net(img_init, img)
    print(target,output)
    # loss = loss_fn(output, target)
    break