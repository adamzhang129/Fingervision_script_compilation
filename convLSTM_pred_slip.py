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


import IPython


class ConvLSTMChained(nn.Module):

    def __init__(self, n_frames_ahead=2, n_frames=10):
        super(ConvLSTMChained, self).__init__()
        self.n_frames_ahead = n_frames_ahead
        self.n_frames = n_frames

        self.pred_net = convLSTMPred(3, 32, 2)
        self.detect_net = convLSTMDetect(3, 64)

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


def load_state_dict(model, path_list):

    model_dict = model.state_dict()
    # for key, value in model_dict.iteritems():
    #     print key

    for type_key, path in path_list.iteritems():
        # print '-----------------------------'
        pretrained_dict = torch.load(path)
        # for key, value in pretrained_dict.iteritems():
        #     print key

        # 1. filter out unnecessary keys
        pretrained_dict = {(type_key + '.' + k): v for k, v in pretrained_dict.items() if (type_key + '.' + k) in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

        # print '-----------------------------'



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

    # print model.state_dict()

    # load pretrained_model_diction
    path_pred = './saved_model/convlstm_frame_predict_20190308_200epochs_3200data_flipped_2f_ahead.pth'
    path_detect = './saved_model/convlstm__model_1layer_augmented_20190308.pth'

    path_dict = {'pred_net': path_pred, 'detect_net': path_detect}

    load_state_dict(model, path_dict)
    
    # IPython.embed()

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

    print ('[ TEST set] Step {}, Loss: {:.6f}, Acc: {:.4f}'.format(
        test_step + 1, test_loss_reduced, test_accuracy))




if __name__ == '__main__':
    _main()