import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable

from convLSTM_slip_detection_1layer import ConvLSTMCell as convLSTMDetect
from convLSTM_frame_pred import ConvLSTMCell as convLSTMPred

# import dataset loader module
from convLSTM_dataset import *

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split


class ConvLSTMChained(nn.Module):

    def __init__(self, n_frames_ahead=2, n_frames=10):
        super(ConvLSTMChained, self).__init__()
        self.n_frames_ahead = n_frames_ahead
        self.n_frames = n_frames

        self.pred_net = convLSTMPred(3, 32, 2)
        self.detect_net = convLSTMDetect(3, 32)

        self.output_list = {'pred': [], 'detect': []}

    def forward(self, t, input, prev): # prev defined as a dict
        prev_p = prev['pred']
        prev_d = prev['detect']
        if t < self.n_frames - self.n_frames_ahead:
            # print '[INFO] forwarding: time frame {}'.format(t)
            out_p, prev_p = self.pred_net(input, prev_p)
            out_d, prev_d = self.detect_net(input, prev_d)

            prev = {'pred': prev_p, 'detect': prev_d}

            self.output_list['pred'].append(out_p)
            self.output_list['detect'].append(out_d)

            # print 'prev state size {}'.format(len(prev['pred']))

        else:

            out_d, prev_d = self.detect_net(self.output_list['pred'][t - self.n_frames_ahead], prev_d)
            prev['detect'] = prev_d
            self.output_list['detect'].append(out_d)

        return out_d, prev




import IPython


def _main():
    """
    Run some basic tests on the API
    """

    # define batch_size, channels, height, width
    batch_size, channels, height, width = 32, 3, 30, 30
    hidden_size = 64 # 64           # hidden state size
    lr = 1e-5     # learning rate
    n_frames = 10           # sequence length
    max_epoch = 30  # number of epochs

    convlstm_dataset = convLSTM_Dataset(dataset_dir='../dataset3/resample_skipping',
                                        n_class=2,
                                        transform=transforms.Compose([
                                            RandomHorizontalFlip(),
                                            RandomVerticalFlip(),
                                            ToTensor(),
                                        ])
                                        )
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

    print('Instantiating model.............')
    model = ConvLSTMChained(n_frames_ahead=2, n_frames=10)
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
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

    # IPython.embed()

    # print('Run for', max_epoch, 'iterations')
    # for epoch in range(0, max_epoch):
    #     loss_train = 0
    #     n_right_train = 0
    #     for step, sample_batched in enumerate(train_dataloader):
    #
    #         model = model.train()
    #
    #         x = sample_batched['frames']
    #         y = sample_batched['target']
    #         x = torch.transpose(x, 0, 1)  # transpose time sequence and batch (N, batch, channel, height, width)
    #         # x = x.type(torch.FloatTensor)
    #         # print x.size()
    #
    #         if torch.cuda.is_available():
    #             # print 'sending input and target to GPU'
    #             x = x.type(torch.cuda.FloatTensor)
    #             y = y.type(torch.cuda.FloatTensor)
    #
    #         state = None
    #         out = None
    #
    #
    #         for t in range(0, n_frames):
    #             # print x[t,0,0,:,:]
    #             out, state = model(x[t], state)
    #             # loss += loss_fn(state[0], y[t])
    #
    #         # out = out.long()
    #         y = y.long()
    #
    #         # print out.size(), y.size()
    #         loss = loss_fn(out, y)
    #         # print(' > Epoch {:2d} loss: {:.7f}'.format((epoch+1), loss.data[0]))
    #
    #         # zero grad parameters
    #         model.zero_grad()
    #
    #         # compute new grad parameters through time!
    #         loss.backward()
    #         optimizer.step()
    #         # learning_rate step against the gradient
    #         # optimizer.step()
    #         #  for p in model.parameters():
    #         #     p.data.sub_(p.grad.data * lr)
    #
    #         loss_train += loss.item()*batch_size
    #         # Compute accuracy
    #
    #         _, argmax = torch.max(out, 1)
    #         # print y, argmax.squeeze()
    #         # accuracy = (y == argmax.squeeze()).float().mean() # accuracy in each batch
    #         n_right_train += sum(y == argmax.squeeze()).item()
    #
    #         if (step + 1) % 50 == 0:
    #             loss_train_reduced = loss_train / (50*batch_size)
    #             train_accuracy = float(n_right_train) / (50*batch_size)
    #             loss_train = 0
    #             n_right_train = 0
    #             print '=================================================================='
    #             print ('[TRAIN set] Epoch {}, Step {}, Loss: {:.6f}, Acc: {:.4f}'
    #                    .format(epoch, step + 1, loss_train_reduced, train_accuracy))
    #
    #
    #
    #             # ================================================================== #
    #             #                        Tensorboard Logging                         #
    #             # ================================================================== #
    #
    #             model = model.eval()
    #
    #             test_loss = 0
    #             n_right = 0
    #             for test_step, test_sample_batched in enumerate(test_dataloader):
    #                 x = test_sample_batched['frames']
    #                 y = test_sample_batched['target']
    #                 x = torch.transpose(x, 0, 1)
    #                 # x = x.type(torch.FloatTensor)
    #
    #                 if torch.cuda.is_available():
    #                     # print 'sending input and target to GPU'
    #                     x = x.type(torch.cuda.FloatTensor)
    #                     y = y.type(torch.cuda.FloatTensor)
    #
    #                 state_test = None
    #                 out_test = None
    #
    #                 for t in range(0, n_frames):
    #                     out_test, state_test = model(x[t], state_test)
    #                     # loss += loss_fn(state[0], y[t])
    #
    #                 # out = out.long()
    #                 y = y.long()
    #
    #                 # print out.size(), y.size()
    #                 test_loss += loss_fn(out_test, y).item() * batch_size
    #
    #                 # Compute accuracy
    #                 _, argmax_test = torch.max(out_test, 1)
    #                 # print argmax_test
    #                 # print y
    #                 n_right += sum(y == argmax_test.squeeze()).item()
    #
    #             # print n_right
    #             test_loss_reduced = test_loss/test_size
    #             test_accuracy = float(n_right)/test_size
    #
    #
    #
    #
    #             # print test_accuracy
    #             print ('[ TEST set] Epoch {}, Step {}, Loss: {:.6f}, Acc: {:.4f}'
    #                    .format(epoch, step + 1, test_loss_reduced, test_accuracy))
    #             # 1. Log scalar values (scalar summary)
    #             info = {'loss': loss_train_reduced, 'accuracy': train_accuracy,
    #                     'test_loss': test_loss_reduced, 'test_accuracy': test_accuracy}
    #
    #             for tag, value in info.items():
    #                 logger.scalar_summary(tag, value, epoch*(train_size/batch_size) + (step + 1))
    #
    #             # 2. Log values and gradients of the parameters (histogram summary)
    #             for tag, value in model.named_parameters():
    #                 tag = tag.replace('.', '/')
    #                 logger.histo_summary(tag, value.data.cpu().numpy(), epoch*(train_size/batch_size) + (step + 1))
    #                 logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch*(train_size/batch_size) + (step + 1))
    #
    #             # 3. Log training images (image summary)
    #             # info = {'images': images.view(-1, 28, 28)[:10].cpu().numpy()}
    #
    #             # for tag, images in info.items():
    #             #     logger.image_summary(tag, images, step + 1)


    import time

    model = model.eval()

    test_loss = 0
    n_right = 0

    start = time.time()
    for test_step, test_sample_batched in enumerate(test_dataloader):
        model.output_list = {'pred': [], 'detect': []}

        x = test_sample_batched['frames']
        y = test_sample_batched['target']
        x = torch.transpose(x, 0, 1)
        # x = x.type(torch.FloatTensor)

        if torch.cuda.is_available():
            # print 'sending input and target to GPU'
            x = x.type(torch.cuda.FloatTensor)
            y = y.type(torch.cuda.FloatTensor)

        prev = {'pred': None, 'detect': None}

        for t in range(0, n_frames):
            out_test,  prev = model(t, x[t], prev)

        y = y.long()

        test_loss += loss_fn(out_test, y).item() * batch_size
        # Compute accuracy
        _, argmax_test = torch.max(out_test, 1)
        # print argmax_test
        # print y
        n_right += sum(y == argmax_test.squeeze()).item()

        # print n_right
    test_loss_reduced = test_loss / test_size
    test_accuracy = float(n_right) / test_size

    print ('[ TEST set] Step {}, Loss: {:.6f}, Acc: {:.4f}'
                               .format(test_step + 1, test_loss_reduced, test_accuracy))


    #     print 'show a batch in test set:'
    #     print y
    #     print y_detect.squeeze()
    #     break
    # print 'one batch inference time:', (time.time() - start)/batch_size
    # save the trained model parameters
    # torch.save(model.state_dict(), './saved_model/convlstm__model_1layer_augmented_20190308.pth') # arbitrary file extension


    # print('Input size:', list(x.data.size()))
    # print('Target size:', list(y.data.size()))



if __name__ == '__main__':
    _main()