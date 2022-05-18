from decimal import DivisionImpossible
from typing import Optional, Union
from numpy import size
import torch
import torch.nn as nn
import torch.distributions as D
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from .functions import *
from .common import *

class NewCNN(nn.Module):

    def __init__(self, in_channels=3, cnn_depth=32, activation=nn.ELU):
        super().__init__()
        self.out_dim = cnn_depth * 32
        kernels = (4, 4, 4, 4)
        stride = 2
        padding = 1
        d = cnn_depth

        self.encoder = nn.Sequential(
            nn.Conv2d(9, d, kernels[0], stride, bias=False),
            nn.LayerNorm([48, 31, 31]),
            activation(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d, in_channels, 4, stride=2, bias=False),
            nn.LayerNorm([3, 64, 64]),
            activation()
        )

        self.x_0 = None
        self.x_1 = None
        self.x_2 = None
        self.x_3 = None
        self.iter = 0
        self.picture_every = 5000

    def forward(self, x: Tensor) -> Tensor:
        if self.x_0 is not None and self.x_0.size() != x.size():
            print("reset history")
            self.x_0 = None

        if self.x_0 is None:
            self.x_0 = x
            self.x_1 = x
            self.x_2 = x
            self.x_3 = x
        else:
            self.x_3 = self.x_2
            self.x_2 = self.x_1
            self.x_1 = self.x_0
            self.x_0 = x

        combined_history = torch.cat((self.x_1, self.x_2, self.x_3), -3)
        combined_history, bd = flatten_batch(combined_history, 3)
        # y = self.model(combined_history)
        # y = unflatten_batch(y, bd)
        # instuff, bd1 = flatten_batch(self.x_3, 3)
        y = self.encoder(combined_history)
        y = self.decoder(y)
        y = unflatten_batch(y, bd)
        # x_1_out, bd1 = flatten_batch(self.x_1, 3)
        # x_2_out, bd2 = flatten_batch(self.x_2, 3)
        # x_3_out, bd3 = flatten_batch(self.x_3, 3)
        # y_1 = self.encoder(x_1_out)
        # y_2 = self.encoder(x_2_out)
        # y_3 = self.encoder(x_3_out)
        # y = torch.stack((y_1, y_2, y_3), 1)
        # # print(f"size y after encoder: {y.size()}")
        # y = self.neuralnetwork(y)
        # # print(f"size y after nn: {y.size()}")
        # y = self.decoder(y)
        # y = unflatten_batch(y, bd1)

        # print(f"size history: {combined_history.size()}")
        # print(f"size x input: {x.size()}")
        # print(f"size y after decoder: {y.size()}")
        # print(f"size y numpy: {np.shape(y.detach().numpy())}")
        # print(f"size y numpy 1 mean: {np.shape(np.mean(y.detach().numpy(), 0))}")

        self.iter += 1
        if self.iter == self.picture_every:
            print("Creating pictures New CNN")
            fig, (ax1, ax2, ax3) = plt.subplots(1,3)
            ax1.imshow(np.clip(x.cpu().detach().numpy().astype('float64')[0][0].transpose((1,2,0)), 0, 1), interpolation='nearest')
            ax1.set_title("Input")
            ax2.imshow(np.clip(y.cpu().detach().numpy().astype('float64')[0][0].transpose((1,2,0)), 0, 1), interpolation='nearest')
            ax2.set_title("CNN_out 1")
            ax3.imshow(np.clip(np.mean(np.mean(y.cpu().detach().numpy().astype('float64'), 0), 0).transpose((1,2,0)), 0, 1), interpolation='nearest')
            ax3.set_title("CNN_out mean")
            plt.savefig('pictures/NewCNN_out.png')
            plt.close(fig)

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(np.clip(self.x_1.cpu().detach().numpy().astype('float64')[0][0].transpose((1,2,0)), 0, 1), interpolation='nearest')
            ax1.set_title("x_1")
            ax2.imshow(np.clip(self.x_2.cpu().detach().numpy().astype('float64')[0][0].transpose((1,2,0)), 0, 1), interpolation='nearest')
            ax2.set_title("x_2")
            ax3.imshow(np.clip(self.x_3.cpu().detach().numpy().astype('float64')[0][0].transpose((1,2,0)), 0, 1), interpolation='nearest')
            ax3.set_title("x_3")
            plt.savefig('pictures/history.png')
            plt.close(fig)
            self.iter = 0
        

        return y