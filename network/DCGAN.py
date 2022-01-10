import torch.nn as nn
from stochman import nnj



class EnergyModel(nn.Module):
    def __init__(self, args, dim=512):
        super().__init__()
        if args.sn==True:
            self.main = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(3, dim // 8, 3, 1, 1)),
                nn.LeakyReLU(0.1, inplace=True),
                nn.utils.spectral_norm(nn.Conv2d(dim // 8, dim // 8, 4, 2, 1)),
                nn.LeakyReLU(0.1, inplace=True),
                nn.utils.spectral_norm(nn.Conv2d(dim // 8, dim // 4, 3, 1, 1)),
                nn.LeakyReLU(0.1, inplace=True),
                nn.utils.spectral_norm(nn.Conv2d(dim // 4, dim // 4, 4, 2, 1)),
                nn.LeakyReLU(0.1, inplace=True),
                nn.utils.spectral_norm(nn.Conv2d(dim // 4, dim // 2, 3, 1, 1)),
                nn.LeakyReLU(0.1, inplace=True),
                nn.utils.spectral_norm(nn.Conv2d(dim // 2, dim // 2, 4, 2, 1)),
                nn.LeakyReLU(0.1, inplace=True),
                nn.utils.spectral_norm(nn.Conv2d(dim // 2, dim, 3, 1, 1)),
                nn.LeakyReLU(0.1, inplace=True)
            )
            self.expand = nn.utils.spectral_norm(nn.Linear(4 * 4 * dim, 1))

        else:
            self.main = nn.Sequential(
                nn.Conv2d(3, dim // 8, 3, 1, 1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(dim // 8, dim // 8, 4, 2, 1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(dim // 8, dim // 4, 3, 1, 1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 4, 2, 1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(dim // 4, dim // 2, 3, 1, 1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(dim // 2, dim // 2, 4, 2, 1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(dim // 2, dim, 3, 1, 1),
                nn.LeakyReLU(0.1, inplace=True),
            )
            self.expand = nn.Linear(4 * 4 * dim, 1)
    def forward(self, x):
        out = self.main(x).view(x.size(0), -1)
        out = self.expand(out).squeeze(-1)
        return out

class Generator(nn.Module):
    def __init__(self, z_dim=128, dim=512):
        super().__init__()

        self.main = nnj.Sequential(
            nnj.Linear(z_dim, 4 * 4 * dim),
            nnj.Reshape(-1,4,4),
            nnj.ConvTranspose2d(dim, dim // 2, 4, 2, 1),
            nnj.BatchNorm2d(dim // 2),
            nnj.ReLU(True),
            nnj.ConvTranspose2d(dim // 2, dim // 4, 4, 2, 1),
            nnj.BatchNorm2d(dim // 4),
            nnj.ReLU(True),
            nnj.ConvTranspose2d(dim // 4, dim // 8, 4, 2, 1),
            nnj.BatchNorm2d(dim // 8),
            nnj.ReLU(True),
            nnj.ConvTranspose2d(dim // 8, 3, 3, 1, 1),
            nnj.Tanh(),
        )
        #self.apply(utils.weights_init)

    def forward(self, z,jacob=False):

        return self.main(z,jacob)