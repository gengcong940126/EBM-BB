import torch.nn as nn
from stochman import nnj
from utils import init_weights
import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation_fn):
        super(GenBlock, self).__init__()
        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError
        self.bn1 = nn.BatchNorm2d(in_features=in_channels,eps=1e-4)
        self.bn2 = nn.BatchNorm2d(in_features=out_channels,eps=1e-4)


        self.upsample0 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2d0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2d1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2d2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x0 = x
        x = self.bn1(x)
        x = self.activation(x)
        x = self.upsample0(x)
        x = self.conv2d1(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv2d2(x)
        x0 = self.upsample1(x0)
        x0 = self.conv2d0(x0)
        out = x + x0
        return out


class Generator(nn.Module):
    """Generator."""
    def __init__(self, args, z_dim,g_conv_dim=64):
        super(Generator, self).__init__()
        g_in_dims_collection = {"32": [g_conv_dim*4, g_conv_dim*4, g_conv_dim*4],
                                "64": [g_conv_dim*16, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2],
                                "128": [g_conv_dim*16, g_conv_dim*16, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2],
                                "256": [g_conv_dim*16, g_conv_dim*16, g_conv_dim*8, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2],
                                "512": [g_conv_dim*16, g_conv_dim*16, g_conv_dim*8, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim]}

        g_out_dims_collection = {"32": [g_conv_dim*4, g_conv_dim*4, g_conv_dim*4],
                                 "64": [g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim],
                                 "128": [g_conv_dim*16, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim],
                                 "256": [g_conv_dim*16, g_conv_dim*8, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim],
                                 "512": [g_conv_dim*16, g_conv_dim*8, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim, g_conv_dim]}
        bottom_collection = {"32": 4, "64": 4, "128": 4, "256": 4, "512": 4}

        self.z_dim = z_dim
        self.in_dims =  g_in_dims_collection[str(args.input_size)]
        self.out_dims = g_out_dims_collection[str(args.input_size)]
        self.bottom = bottom_collection[str(args.input_size)]
        self.linear0 = nn.Linear(in_features=self.z_dim, out_features=self.in_dims[0]*self.bottom*self.bottom)

        self.blocks = []
        for index in range(len(self.in_dims)):
            self.blocks += [[GenBlock(in_channels=self.in_dims[index],
                                          out_channels=self.out_dims[index],
                                          activation_fn='ReLU')]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.bn4 = nn.BatchNorm2d(in_features=self.out_dims[-1], eps=1e-4)

        self.activation = nn.ReLU(inplace=True)

        self.conv2d5 = nn.Conv2d(in_channels=self.out_dims[-1], out_channels=3, kernel_size=3, stride=1, padding=1)

        self.tanh = nn.Tanh()

        #init_weights(self.modules, initialize)

    def forward(self, z):
        act = self.linear0(z)
        act = act.view(-1, self.in_dims[0], self.bottom, self.bottom)
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                act = block(act)
        act = self.bn4(act)
        act = self.activation(act)
        act = self.conv2d5(act)
        out = self.tanh(act)
        return out


class DiscOptBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_spectral_norm, bn,activation_fn):
        super(DiscOptBlock, self).__init__()
        self.d_spectral_norm = d_spectral_norm
        self.bn=bn

        if d_spectral_norm:
            self.conv2d0 = spectral_norm(nn.Conv2d(in_channels=in_channels,
                        out_channels=out_channels, kernel_size=1, stride=1, padding=0),eps=1e-6)
            self.conv2d1 = spectral_norm(nn.Conv2d(in_channels=in_channels,
                        out_channels=out_channels, kernel_size=3, stride=1, padding=1),eps=1e-6)
            self.conv2d2 = spectral_norm(nn.Conv2d(in_channels=out_channels,
                        out_channels=out_channels, kernel_size=3, stride=1, padding=1),eps=1e-6)
        else:
            self.conv2d0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.conv2d1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

            self.bn0 = nn.BatchNorm2d(in_features=in_channels,momentum=0.9,eps=1e-4)
            self.bn1 = nn.BatchNorm2d(in_features=out_channels,momentum=0.9,eps=1e-4)

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        self.average_pooling = nn.AvgPool2d(2)


    def forward(self, x):
        x0 = x

        x = self.conv2d1(x)
        if self.bn is True:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2d2(x)
        x = self.average_pooling(x)

        x0 = self.average_pooling(x0)
        if self.bn is True:
            x0 = self.bn0(x0)
        x0 = self.conv2d0(x0)

        out = x + x0
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_spectral_norm, bn,activation_fn, downsample=True):
        super(DiscBlock, self).__init__()
        self.d_spectral_norm = d_spectral_norm
        self.bn=bn
        self.downsample = downsample

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True

        if d_spectral_norm:
            if self.ch_mismatch or downsample:
                self.conv2d0 = spectral_norm(nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels, kernel_size=1, stride=1, padding=0),eps=1e-6)
            self.conv2d1 = spectral_norm(nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels, kernel_size=3, stride=1, padding=1),eps=1e-6)
            self.conv2d2 = spectral_norm(nn.Conv2d(in_channels=out_channels,
                            out_channels=out_channels, kernel_size=3, stride=1, padding=1),eps=1e-6)
        else:
            if self.ch_mismatch or downsample:
                self.conv2d0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.conv2d1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

            if self.ch_mismatch or downsample:
                self.bn0 = nn.BatchNorm2d(in_features=in_channels,momentum=0.9,eps=1e-4)
            self.bn1 = nn.BatchNorm2d(in_features=in_channels,momentum=0.9,eps=1e-4)
            self.bn2 = nn.BatchNorm2d(in_features=out_channels,momentum=0.9,eps=1e-4)

        self.average_pooling = nn.AvgPool2d(2)


    def forward(self, x):
        x0 = x
        if self.bn is True:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2d1(x)
        if self.bn is True:
            x = self.bn2(x)
        x = self.activation(x)
        x = self.conv2d2(x)
        if self.downsample:
            x = self.average_pooling(x)

        if self.downsample or self.ch_mismatch:
            if self.bn is True:
                x0 = self.bn0(x0)
            x0 = self.conv2d0(x0)
            if self.downsample:
                x0 = self.average_pooling(x0)

        out = x + x0
        return out


class EnergyModel(nn.Module):
    """Energy function."""
    def __init__(self, args, d_conv_dim=64):
        super(EnergyModel, self).__init__()
        d_in_dims_collection = {"32": [3] + [d_conv_dim*2, d_conv_dim*2, d_conv_dim*2],
                                "64": [3] +[d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8],
                                "128": [3] +[d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*16],
                                "256": [3] +[d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*8, d_conv_dim*16],
                                "512": [3] +[d_conv_dim, d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*8, d_conv_dim*16]}

        d_out_dims_collection = {"32": [d_conv_dim*2, d_conv_dim*2, d_conv_dim*2, d_conv_dim*2],
                                 "64": [d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*16],
                                 "128": [d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*16, d_conv_dim*16],
                                 "256": [d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*8, d_conv_dim*16, d_conv_dim*16],
                                 "512": [d_conv_dim, d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*8, d_conv_dim*16, d_conv_dim*16]}

        d_down = {"32": [True, True, False, False],
                  "64": [True, True, True, True, False],
                  "128": [True, True, True, True, True, False],
                  "256": [True, True, True, True, True, True, False],
                  "512": [True, True, True, True, True, True, True, False]}


        self.in_dims  = d_in_dims_collection[str(args.input_size)]
        self.out_dims = d_out_dims_collection[str(args.input_size)]
        down = d_down[str(args.input_size)]

        self.blocks = []
        for index in range(len(self.in_dims)):
            if index == 0:
                self.blocks += [[DiscOptBlock(in_channels=self.in_dims[index],
                                              out_channels=self.out_dims[index],
                                              d_spectral_norm=args.sn,
                                              bn=False,
                                              activation_fn='Leaky_ReLU')]]
            else:
                self.blocks += [[DiscBlock(in_channels=self.in_dims[index],
                                           out_channels=self.out_dims[index],
                                           d_spectral_norm=args.sn,
                                           bn=False,
                                           activation_fn='Leaky_ReLU',
                                           downsample=down[index])]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if args.sn:
            self.linear1 = spectral_norm(nn.Linear(in_features=self.out_dims[-1], out_features=1), eps=1e-6)

        else:
            self.linear1 = nn.Linear(in_features=self.out_dims[-1], out_features=1)

        #init_weights(self.modules, initialize)


    def forward(self, x):
        h = x
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        h = self.activation(h)
        h = torch.sum(h, dim=[2,3])
        authen_output = torch.squeeze(self.linear1(h))
        return authen_output

