from convLSTM_slip_detection_3 import ConvLSTMCell


import torch
import tensorflow as tf


channels = 3
hidden_size = 64

# model = ConvLSTMCell(channels, hidden_size)
#
#
# def get_n_params(model):
#     pp=0
#     for p in list(model.parameters()):
#         nn=1
#         for s in list(p.size()):
#             nn = nn*s
#         pp += nn
#     return pp
#
# print 'number of model parameters:{}'.format(get_n_params(model))


# model.load_state_dict(torch.load('./saved_model/convlstm_model.pth'))
test_accuracy = []
test_loss = []
accuracy = []
loss = []

for summary in tf.train.summary_iterator('logs/events.out.tfevents.1536734255.adam-PC'):
    for v in summary.summary.value:
        if v.tag == 'test_accuracy':
            test_accuracy.append(v.simple_value)
        elif v.tag == 'test_loss':
            test_loss.append(v.simple_value)
        elif v.tag == 'accuracy':
            accuracy.append(v.simple_value)
        elif v.tag == 'loss':
            loss.append(v.simple_value)

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.ndimage

from matplotlib.font_manager import FontProperties


font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)


# accuracy = scipy.ndimage.gaussian_filter(accuracy, 2)
# loss = scipy.ndimage.gaussian_filter(loss, 2)

# plot training process
step = range(0, len(accuracy))
step = np.array(step)*50

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(step, accuracy)
# ax.plot(step, loss)
# ax.plot(step, test_accuracy)
# ax.plot(step, test_loss)
# plt.show()

# Create some mock data
t = step


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Steps',fontsize=16)
ax1.set_ylabel('Accuracy', fontsize=16, color=color)
ax1.plot(t, accuracy, linewidth=2, label='Train Accuracy', color=color)
ax1.plot(t, test_accuracy, linewidth=2, label='Test Accuracy', color='tab:green')

ax1.tick_params(axis='y', labelcolor=color)



plt.grid()
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Loss', fontsize=16, color=color)  # we already handled the x-label with ax1
ax2.plot(t, loss, linewidth=2, label='Train Loss', color=color)
ax2.plot(t, test_loss, linewidth=2, label='Test Loss', color='tab:pink')
ax2.tick_params(axis='y', labelcolor=color)


ax1.legend(loc=(0.6,0.70))
ax2.legend(loc=(0.6, 0.54))
fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show()