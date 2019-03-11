import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
import torch


# Define some constants
KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size, n_frames_ahead):
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.n_frames_ahead = n_frames_ahead
        self.hidden_size = hidden_size
        self.Gates_layer1 = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, KERNEL_SIZE, padding=PADDING)

        self.Gates_layer2 = nn.Conv2d(2*hidden_size, 4*hidden_size, KERNEL_SIZE, padding=PADDING)

        self.height, self.width = 30, 30

        self.Shrink = nn.Conv2d(hidden_size, self.input_size, KERNEL_SIZE, padding=PADDING)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # print spatial_size

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size1 = [batch_size, self.hidden_size] + list(spatial_size)  # (B,Hidden_size, H, W)
            prev_state1 = (
                Variable(torch.zeros(state_size1)).type(torch.cuda.FloatTensor),
                Variable(torch.zeros(state_size1)).type(torch.cuda.FloatTensor)
            )  # list of h[t-1] and C[t-1]: both of size [batch_size, hidden_size, D, D]

            # =======layer2 lstm previous states
            state_size2 = [batch_size, self.hidden_size] + list(spatial_size)

            prev_state2 = (
                Variable(torch.zeros(state_size2)).type(torch.cuda.FloatTensor),
                Variable(torch.zeros(state_size2)).type(torch.cuda.FloatTensor)
            )  # list of h[t-1] and C[t-1]: both of size [batch_size, hidden_size, D, D]

            prev_state = (prev_state1, prev_state2)

        prev_state1, prev_state2 = prev_state
        prev_hidden1, prev_cell1 = prev_state1
        prev_hidden2, prev_cell2 = prev_state2

        # data size is [batch, channel, height, width]
        # print input_.type(), prev_hidden.type()
        stacked_inputs = torch.cat((input_, prev_hidden1), 1)  # concat x[t] with h[t-1]
        gates = self.Gates_layer1(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = f.tanh(cell_gate)
        # print cell_gate.shape

        # compute current cell and hidden state
        cell1 = (remember_gate * prev_cell1) + (in_gate * cell_gate)
        hidden1 = out_gate * f.tanh(cell1)

        # print hidden.size()
        # conv2 = self.Conv(hidden)
        # flat = conv2.view(-1, conv2.size(1) * conv2.size(2) * conv2.size(3))

        # =============layer 2 gates operation =================================
        # feed hidden state from layer 1 to layer 2 as input
        stacked_inputs2 = torch.cat((hidden1, prev_hidden2), 1)  # concat x[t] with h[t-1]
        gates2 = self.Gates_layer2(stacked_inputs2)

        # chunk across channel dimension
        in_gate2, remember_gate2, out_gate2, cell_gate2 = gates2.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate2 = torch.sigmoid(in_gate2)
        remember_gate2 = torch.sigmoid(remember_gate2)
        out_gate2 = torch.sigmoid(out_gate2)

        # apply tanh non linearity
        cell_gate2 = torch.tanh(cell_gate2)
        # print cell_gate.shape

        # compute current cell and hidden state
        cell2 = (remember_gate2 * prev_cell2) + (in_gate2 * cell_gate2)
        hidden2 = out_gate2 * f.tanh(cell2)

        # flat = hidden2.view(-1, hidden2.size(1)*hidden2.size(2)*hidden2.size(3))
        # # print flat.size()
        # out = self.linear(flat)
        # out = self.dropout(out)

        out = self.Shrink(hidden2)

        # print out.shape

        # print out.shape
        # out = out.view(-1, self.n_frames_ahead, self.input_size, self.height, self.width)
        # print out.shape
        # out = torch.transpose(out, 0, 1)

        return out, ((hidden1, cell1), (hidden2, cell2))



from convLSTM_dataset import *
from torch.utils.data import DataLoader

from torch.utils.data.dataset import random_split

# from logger import Logger
# logger = Logger('./logs')

import IPython

from torch.utils.data.sampler import SubsetRandomSampler


def random_split(dataset, train_ratio=0.9, shuffle_dataset=True):
    random_seed = 41
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    # print indices
    split = int(np.floor(train_ratio * dataset_size))
    # print split
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    # print indices[0:10]
    train_indices, test_indices = indices[:split], indices[split:]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    return train_sampler, test_sampler


import time

def _main():
    """
    Run some basic tests on the API
    """

    # define batch_size, channels, height, width
    batch_size, channels, height, width = 32, 3, 30, 30
    hidden_size = 32 # 64           # hidden state size
    lr = 1e-5     # learning rate
    max_epoch = 1  # number of epochs
    # n_frames = 8     # sequence length
    #
    #
    # n_frames_ahead = 10 - n_frames

    convlstm_dataset = convLSTM_Dataset(dataset_dir='../dataset3/resample_skipping',
                                        n_class=2,
                                        transform=transforms.Compose([
                                            RandomHorizontalFlip(),
                                            RandomVerticalFlip(),
                                            ToTensor()])
                                        )
    # convlstm_dataset = convLSTM_tdiff_Dataset(dataset_dir='../dataset3/resample_skipping',
    #                                           n_class=2,
    #                                           transform=transforms.Compose([
    #                                               ToTensor()])
    #                                           )
    # train_ratio = 0.9
    # train_size = int(train_ratio*len(convlstm_dataset))
    # test_size = len(convlstm_dataset) - train_size
    #
    # train_dataset, test_dataset = random_split(convlstm_dataset, [train_size, test_size])

    train_sampler, test_sampler = random_split(convlstm_dataset, train_ratio=0.9)

    train_dataloader = DataLoader(convlstm_dataset, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=4)
    test_dataloader = DataLoader(convlstm_dataset, batch_size=batch_size, sampler=test_sampler,
                                 num_workers=4)
    # IPython.embed()
    # set manual seed
    # torch.manual_seed(0)

    # train with different values of n_frames_ahead to see the performance
    for n_frames_ahead in range(1, 6):
        n_frames = 10 - n_frames_ahead

        print '[Train with n_frames_ahead = {}]'.format(n_frames_ahead)
        print('Instantiate model')
        model = ConvLSTMCell(channels, hidden_size, n_frames_ahead)
        print(repr(model))

        if torch.cuda.is_available():
            # print 'sending model to GPU'
            model = model.cuda()

        print('Create input and target Variables')
        x = Variable(torch.rand(n_frames, batch_size, channels, height, width))
        # y = Variable(torch.randn(T, b, d, h, w))
        y = Variable(torch.rand(batch_size))

        print('Create a MSE criterion')
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.05)

        model.train()

        # -----------------------------------------------------------
        print('Start the training, Running for', max_epoch, 'epochs')
        for epoch in range(0, max_epoch):
            loss_train = 0
            n_right_train = 0

            for step, sample_batched in enumerate(train_dataloader):
                model = model.train()
                loss = 0

                frames = sample_batched['frames']

                # y = sample_batched['target']
                # transpose time sequence and batch (sequence, batch, channel, height, width)
                frames = torch.transpose(frames, 0, 1)

                x = frames[:n_frames]
                y = frames[n_frames:]
                # IPython.embed()
                # x = x.type(torch.FloatTensor)
                # print x.size()

                if torch.cuda.is_available():
                    # print 'sending input and target to GPU'
                    x = x.type(torch.cuda.FloatTensor)
                    y = y.type(torch.cuda.FloatTensor)

                state = None
                out = None

                # IPython.embed()
                for t in range(0, n_frames):
                    # print x[t,0,0,:,:]
                    out, state = model(x[t], state)
                    if t in range(0, n_frames)[-n_frames_ahead:]:
                        # IPython.embed()
                        loss += loss_fn(out, y[n_frames_ahead - (n_frames - t)])

                # reduce loss to be loss on single frame discrepancy/loss

                # zero grad parameters
                model.zero_grad()

                # compute new grad parameters through time!
                loss.backward()
                optimizer.step()

                loss_train += loss.item() * batch_size / n_frames_ahead


                Step = 20
                if (step + 1) % Step == 0:
                    loss_train_reduced = loss_train / (Step * batch_size)
                    loss_train = 0.
                    print '=================================================================='
                    print ('[TRAIN set] Epoch {}, Step {}, Average Loss (every 20 steps): {:.6f}'
                           .format(epoch, step + 1, loss_train_reduced))


        model_path = './saved_model/convlstm_frame_predict_20190311_200epochs_3200data_flipped_{}f_ahead.pth'\
            .format(n_frames_ahead)
        # torch.save(model.state_dict(), model_path)




        print 'Starting the evaluation over test set.....'
        model = model.eval()

        start = time.time()
        loss_test = 0
        for test_step, test_sample_batched in enumerate(test_dataloader):
            loss = 0.

            frames = test_sample_batched['frames']
            # y = test_sample_batched['target']
            frames = torch.transpose(frames, 0, 1)
            # x = x.type(torch.FloatTensor)
            x = frames[:n_frames]
            y = frames[n_frames:]
            # IPython.embed()

            if torch.cuda.is_available():
                # print 'sending input and target to GPU'
                x = x.type(torch.cuda.FloatTensor)
                y = y.type(torch.cuda.FloatTensor)

            state_test = None
            out_test = None

            for t in range(0, n_frames):
                out_test, state_test = model(x[t], state_test)
                if t in range(0, n_frames)[-n_frames_ahead:]:
                    # IPython.embed()
                    loss += loss_fn(out_test, y[n_frames_ahead - (n_frames - t)])

            loss_test += loss.item() * batch_size / n_frames_ahead

        # ---------------------------------
        loss_test_reduced = loss_test / len(test_sampler)
        print ('[TEST set] Average Loss (over all set): {:.6f}'
               .format(loss_test_reduced))

        gt = y.squeeze()[1][0]
        gt = gt.cpu().detach().numpy()
        out_single = out_test[0].cpu().detach().numpy()



        # IPython.embed()
        show_two_img(gt, out_single)


def show_two_img(a, b):
    # images are of size: 3X30X30
    plt.figure(figsize=(10, 6))

    for j, x in enumerate([a, b]):
        pic = x[0]
        plt.subplot(3, 2, j + 1)
        # IPython.embed()
        plt.imshow(pic)
        plt.axis('off')
        pic = x[1]
        plt.subplot(3, 2, 2 + j + 1)
        plt.imshow(pic)
        plt.axis('off')
        pic = x[2]
        plt.subplot(3, 2, 2 * 2 + j + 1)
        plt.imshow(pic)
        plt.axis('off')

    plt.show()






if __name__ == '__main__':
    _main()
