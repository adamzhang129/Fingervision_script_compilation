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

    def __init__(self, input_size, hidden_size):
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates_layer1 = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, KERNEL_SIZE, padding=PADDING)

        self.Gates_layer2 = nn.Conv2d(2*hidden_size, 4*hidden_size, KERNEL_SIZE, padding=PADDING)

        # self.Gates_layer2 = nn.Sequential(
        #     nn.Conv2d(4 * hidden_size, 4 * hidden_size, KERNEL_SIZE, padding=PADDING),
        #     nn.ReLU())
        # self.Gates_layer3 = nn.Sequential(
        #     nn.Conv2d(4 * hidden_size, 4 * hidden_size, KERNEL_SIZE, padding=PADDING),
        #     nn.ReLU())
        # self.Gates_layer4 = nn.Sequential(
        #     nn.Conv2d(4 * hidden_size, 4 * hidden_size, KERNEL_SIZE, padding=PADDING),
        #     nn.ReLU())

        # ker2_size = int(0.25 * hidden_size)
        # self.Conv = nn.Conv2d(hidden_size, ker2_size, KERNEL_SIZE, padding=PADDING)



        self.height, self.width = 30, 30

        self.linear = nn.Linear(self.hidden_size*self.height*self.width, 2)
        self.dropout = nn.Dropout(0.3)

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

        # =============layer 2 gates operation =============
        stacked_inputs2 = torch.cat((hidden1, prev_hidden2), 1)  # concat x[t] with h[t-1]
        gates2 = self.Gates_layer2(stacked_inputs2)

        # chunk across channel dimension
        in_gate2, remember_gate2, out_gate2, cell_gate2 = gates2.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate2 = f.sigmoid(in_gate2)
        remember_gate2 = f.sigmoid(remember_gate2)
        out_gate2 = f.sigmoid(out_gate2)

        # apply tanh non linearity
        cell_gate2 = f.tanh(cell_gate2)
        # print cell_gate.shape

        # compute current cell and hidden state
        cell2 = (remember_gate2 * prev_cell2) + (in_gate2 * cell_gate2)
        hidden2 = out_gate2 * f.tanh(cell2)

        flat = hidden2.view(-1, hidden2.size(1)*hidden2.size(2)*hidden2.size(3))
        # print flat.size()
        out = self.linear(flat)
        out = self.dropout(out)

        return out, ((hidden1, cell1), (hidden2, cell2))



from convLSTM_dataset import *
from torch.utils.data import DataLoader

from torch.utils.data.dataset import random_split

from logger import Logger
logger = Logger('./logs')
def _main():
    """
    Run some basic tests on the API
    """

    # define batch_size, channels, height, width
    batch_size, channels, height, width = 32, 3, 30, 30
    hidden_size = 32 # 64           # hidden state size
    lr = 1e-5     # learning rate
    n_frames = 10           # sequence length
    max_epoch = 30  # number of epochs

    convlstm_dataset = convLSTM_Dataset(dataset_dir='../dataset3/resample_skipping',
                                        n_class=2,
                                        transform=transforms.Compose([
                                            ToTensor()])
                                        )
    # convlstm_dataset = convLSTM_tdiff_Dataset(dataset_dir='../dataset3/resample_skipping',
    #                                           n_class=2,
    #                                           transform=transforms.Compose([
    #                                               ToTensor()])
    #                                           )
    train_ratio = 0.9
    train_size = int(train_ratio*len(convlstm_dataset))
    test_size = len(convlstm_dataset) - train_size

    train_dataset, test_dataset = random_split(convlstm_dataset, [train_size, test_size])


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=4)

    # set manual seed
    # torch.manual_seed(0)

    print('Instantiate model')
    model = ConvLSTMCell(channels, hidden_size)
    print(repr(model))

    if torch.cuda.is_available():
        # print 'sending model to GPU'
        model = model.cuda()

    print('Create input and target Variables')
    x = Variable(torch.rand(n_frames, batch_size, channels, height, width))
    # y = Variable(torch.randn(T, b, d, h, w))
    y = Variable(torch.rand(batch_size))

    print('Create a MSE criterion')
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)



    print('Run for', max_epoch, 'iterations')
    for epoch in range(0, max_epoch):
        loss_train = 0
        n_right_train = 0
        for step, sample_batched in enumerate(train_dataloader):
            x = sample_batched['frames']
            y = sample_batched['target']
            x = torch.transpose(x, 0, 1)  # transpose time sequence and batch (N, batch, channel, height, width)
            # x = x.type(torch.FloatTensor)
            # print x.size()

            if torch.cuda.is_available():
                # print 'sending input and target to GPU'
                x = x.type(torch.cuda.FloatTensor)
                y = y.type(torch.cuda.FloatTensor)

            state = None
            out = None


            for t in range(0, n_frames):
                # print x[t,0,0,:,:]
                out, state = model(x[t], state)
                # loss += loss_fn(state[0], y[t])

            # out = out.long()
            y = y.long()

            # print out.size(), y.size()
            loss = loss_fn(out, y)
            # print(' > Epoch {:2d} loss: {:.7f}'.format((epoch+1), loss.data[0]))

            # zero grad parameters
            model.zero_grad()

            # compute new grad parameters through time!
            loss.backward()
            optimizer.step()
            # learning_rate step against the gradient
            # optimizer.step()
            #  for p in model.parameters():
            #     p.data.sub_(p.grad.data * lr)

            loss_train += loss.item()*batch_size
            # Compute accuracy

            _, argmax = torch.max(out, 1)
            # print y, argmax.squeeze()
            # accuracy = (y == argmax.squeeze()).float().mean() # accuracy in each batch
            n_right_train += sum(y == argmax.squeeze()).item()

            if (step + 1) % 50 == 0:
                loss_train_reduced = loss_train / (50*batch_size)
                train_accuracy = float(n_right_train) / (50*batch_size)
                loss_train = 0
                n_right_train = 0
                print '=================================================================='
                print ('[TRAIN set] Epoch {}, Step {}, Loss: {:.6f}, Acc: {:.4f}'
                       .format(epoch, step + 1, loss_train_reduced, train_accuracy))



                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #

                test_loss = 0
                n_right = 0
                for test_step, test_sample_batched in enumerate(test_dataloader):
                    x = test_sample_batched['frames']
                    y = test_sample_batched['target']
                    x = torch.transpose(x, 0, 1)
                    # x = x.type(torch.FloatTensor)

                    if torch.cuda.is_available():
                        # print 'sending input and target to GPU'
                        x = x.type(torch.cuda.FloatTensor)
                        y = y.type(torch.cuda.FloatTensor)

                    state_test = None
                    out_test = None

                    for t in range(0, n_frames):
                        out_test, state_test = model(x[t], state_test)
                        # loss += loss_fn(state[0], y[t])

                    # out = out.long()
                    y = y.long()

                    # print out.size(), y.size()
                    test_loss += loss_fn(out_test, y).item() * batch_size

                    # Compute accuracy
                    _, argmax_test = torch.max(out_test, 1)
                    # print argmax_test
                    # print y
                    n_right += sum(y == argmax_test.squeeze()).item()

                # print n_right
                test_loss_reduced = test_loss/test_size
                test_accuracy = float(n_right)/test_size




                # print test_accuracy
                print ('[ TEST set] Epoch {}, Step {}, Loss: {:.6f}, Acc: {:.4f}'
                       .format(epoch, step + 1, test_loss_reduced, test_accuracy))
                # 1. Log scalar values (scalar summary)
                info = {'loss': loss_train_reduced, 'accuracy': train_accuracy,
                        'test_loss': test_loss_reduced, 'test_accuracy': test_accuracy}

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch*(train_size/batch_size) + (step + 1))

                # 2. Log values and gradients of the parameters (histogram summary)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), epoch*(train_size/batch_size) + (step + 1))
                    logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch*(train_size/batch_size) + (step + 1))

                # 3. Log training images (image summary)
                # info = {'images': images.view(-1, 28, 28)[:10].cpu().numpy()}

                # for tag, images in info.items():
                #     logger.image_summary(tag, images, step + 1)


    import time

    start = time.time()
    for test_step, test_sample_batched in enumerate(test_dataloader):
        x = test_sample_batched['frames']
        y = test_sample_batched['target']
        x = torch.transpose(x, 0, 1)
        # x = x.type(torch.FloatTensor)

        if torch.cuda.is_available():
            # print 'sending input and target to GPU'
            x = x.type(torch.cuda.FloatTensor)
            y = y.type(torch.cuda.FloatTensor)

        state_test = None
        out_test = None

        for t in range(0, n_frames):
            out_test, state_test = model(x[t], state_test)

        _, argmax_test = torch.max(out_test, 1)

        print 'show a batch in test set:'
        print y
        print argmax_test.squeeze()
        break
    print 'one batch inference time:', (time.time() - start)/batch_size
    # save the trained model parameters
    torch.save(model.state_dict(), './saved_model/convlstm_model_2layers_20190301.pth') # arbitrary file extension


    print('Input size:', list(x.data.size()))
    print('Target size:', list(y.data.size()))
    # print('Last hidden state size:', list(state[0].size()))


if __name__ == '__main__':
    _main()
