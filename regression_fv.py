import torch
import torch.nn as nn
import torch.optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.optim import Adam, SGD
from torch.autograd import Variable
import numpy as np

from torch.nn import init

import torchvision.models as models

import matplotlib.pyplot as plt
from FvDataset_vec import *
import os

EPOCH = 10
BATCH_SIZE = 32
LR = 0.0001
D_out = 4
D_in = 3600
D_h = 512

print os.getcwd()


fv_dataset = FvDataset(csv_in_dir='../dataset2/interpolation/vectors',
                           csv_out='../dataset2/wrench_loc_normalized.csv',
                           n_feature=D_out +2,
                           transform=transforms.Compose([
                               ToTensor()])
                       )

train_loader = DataLoader(dataset=fv_dataset, batch_size=BATCH_SIZE, shuffle=True)



# define regression network

class FvRegNet(nn.Module):

    def __init__(self, D_in, D_out, D_h):

        super(FvRegNet, self).__init__()
        # Calling Super Class's constructor
        self.hidden = nn.Linear(D_in, D_h)
        self.hidden1 = nn.Linear(D_h, 32)
        # self.hidden2 = nn.Linear(256, 64)
        # self.hidden3 = nn.Linear(64, 32)
        self.sigmoid = nn.Sigmoid()
        self.output = nn.Linear(32, D_out)
        # nn.linear is defined in nn.Module

    def forward(self, x):
        # Here the forward pass is simply a linear function
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden1(x))

        # x = F.relu(self.sigmoid(x))
        out = self.output(x)

        return out




device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('CUDA available?: ' + str(torch.cuda.is_available()))
net = FvRegNet(D_in=D_in, D_out=D_out, D_h=D_h).cuda()
print(net)



optimizer = torch.optim.Adam(net.parameters(), lr=LR , weight_decay=0.99)
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
        feature_in = sample_batched['feature_in'].float().cuda()
        # need to add .float() to fix bug: https://github.com/pytorch/pytorch/issues/2138
        feature_out = sample_batched['feature_out'].float().cuda()
        # print feature_in.shape
        # print feature_out

        # print feature_in.shape, feature_out.shape

        output = net(feature_in)

        loss = loss_fn(output, feature_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if step % 50 == 0:
        print('epoch: ', epoch, 'step:', step, 'loss: ', loss.cpu().data.numpy())
