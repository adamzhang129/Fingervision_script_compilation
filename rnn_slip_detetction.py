import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

torch.manual_seed(1)

#
num_epochs = 1
batch_size = 64

TIME_STEP = 28        # rnn time step,
INPUT_SIZE = 28       # rnn input size,
HIDDEN_SIZE = 32
NUM_LAYERS = 2
NUM_CLASSES = 10
BIDIRECTIONAL = True  #

learning_rate = 0.01

train_data = dsets.MNIST(
    root='./mnist/',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

#
# print(train_data.train_data.size())    # (60000, 28, 28)
# print(train_data.train_labels.size())  # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = dsets.MNIST(
    root='./mnist/',
    train=False,
    transform=transforms.ToTensor()
)
# test_x shape (-1, 28, 28) value in range(0,1)
test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor) / 255.
test_y = test_data.test_labels
if torch.cuda.is_available():
    test_x = test_x.cuda()
    test_y = test_y.cuda()


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,     #
            num_layers=num_layers,       #
            batch_first=True,  #  batch, :(batch, time_step, input_size)
            bidirectional=bidirectional  #
        )

        self.out = nn.Linear(hidden_size * 2, num_classes) if bidirectional else nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, time_step, input_size)
        # r_out: (batch, time_step, output_size)
        # h_n: (n_layers, batch, hidden_size)
        # h_c: (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)  # None means to initiate hidden variables with zeros

        # (batch, -1, output_size) only output last frame output
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    bidirectional=BIDIRECTIONAL
)
print(rnn)

if torch.cuda.is_available():
    print 'using GPU'
    rnn = rnn.cuda()

optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()

for num_epochs in range(num_epochs):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1, 28, 28))  # reshape x to (batch, time_step, input_size)
        b_y = Variable(y)
        if torch.cuda.is_available():
            b_x = b_x.cuda()
            b_y = b_y.cuda()

        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss

        if step % 50 == 0:
            test_output = rnn(test_x)  # (samples, time_step, input_size)
            pred_y = torch.max(test_output, dim=1)[1].data.squeeze()
            # print sum(pred_y == test_y), test_y.size(0)
            accuracy = float(sum(pred_y == test_y))/test_y.size(0)
            # print accuracy
            print('num_epochs: ', num_epochs, '| train loss: %.5f' % loss.data[0], '| test accuracy: %.4f' % accuracy)

#
test_output = rnn(test_x[:20].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
print(pred_y)
print(test_y[:20].cpu().numpy().squeeze())

print(test_data.test_data.size())    # (60000, 28, 28)
print(test_data.test_labels.size())  # (60000)
plt.imshow(test_data.test_data[0].numpy(), cmap='gray')
plt.title('%i' % test_data.test_labels[0])
plt.show()


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

print 'number of model parameters:{}'.format(get_n_params(rnn))


import torchvision.models as models

squeezenet = models.squeezenet1_1(pretrained=True)

print get_n_params(squeezenet)