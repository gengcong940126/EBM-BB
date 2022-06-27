import torch
import torch.nn as nn
import json
import time
import utils
from data import data_process
import torch.nn.functional as F
import sklearn.datasets
import collections
import os, argparse
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import numpy as np
import torchplot as plt
import torch.distributions as td
from stochman import nnj

@utils.register_model(name='EBM_0GP')
class EBM_0GP(nn.Module):
    def __init__(self, args, layers, device='cpu'):
        super(EBM_0GP, self).__init__()

        self.device = device
        self.D = int(layers[0])  # Dimension of x
        self.d = int(layers[-1])  # Dimension of z
        self.h = layers[1:-1]  # Dimension of hidden layers
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.model_name = args.EBM_type
        self.sample_z_ = torch.randn((self.batch_size, self.d)).to(self.device)

        self.disc = Disc()
        self.gen = gen()
        self.to(self.device)

    def trainer(self, iters=50):

        optimizer_d = torch.optim.Adam(self.disc.parameters(), lr=args.lrd,
                                       betas=(0.0, 0.9), eps=1e-6)
        optimizer_g = torch.optim.Adam(self.gen.parameters(), lr=args.lrg,
                                       betas=(0.0, 0.9), eps=1e-6)
        sum_loss_d = 0
        sum_loss_g = 0
        writer = SummaryWriter(log_dir=self.log_dir)
        for iteration in range(iters):
            epoch = iteration
            e_costs = []
            g_costs = []
           #energy function
            for i in range(args.energy_model_iters):
                data = data_process(args.dataset, args.batch_size)
                data = data.to(self.device)
                data.requires_grad_()
                z_train = torch.randn((data.shape[0], self.d)).to(self.device)
                G_z = self.gen(z_train)
                d_real = self.disc(data)
                d_fake = self.disc(G_z.detach())
                gradients = torch.autograd.grad(
                    outputs=d_real,
                    inputs=data,
                    grad_outputs=torch.ones_like(d_real),
                    allow_unused=True,
                    create_graph=True,
                )[0]
                gradients = gradients.flatten(start_dim=1)
                gp_loss = (gradients.norm(2, dim=1) ** 2).mean()
                D_loss = (d_real - d_fake).mean() + gp_loss * args.gp_weight

                e_costs.append([d_real.mean().item(), d_fake.mean().item(), D_loss.item(), gp_loss.item()])
                optimizer_d.zero_grad()
                D_loss.backward()
                optimizer_d.step()
            d_real_mean, d_fake_mean, D_loss_mean, gp_loss_mean = np.mean(e_costs[-args.energy_model_iters:], 0)
            sum_loss_d += D_loss_mean.item() * len(data)

            # generator
            for i in range(args.generator_iters):
                z_train = torch.randn((data.shape[0], self.d)).to(self.device)
                z_train.requires_grad_()

                if args.train_mode == 'acc':
                    fake = self.gen(z_train)
                    d_fake_g = self.disc(fake)
                    H = self.compute_entropy_acc(z_train)
                else:
                    fake = self.gen(z_train)
                    d_fake_g = self.disc(fake)
                    H = self.compute_entropy_mins_eval(z_train)
                g_loss = d_fake_g.mean() - H*args.H_weight
                g_costs.append([d_fake_g.mean().item(), g_loss.item(), H.mean().item()])
                optimizer_g.zero_grad()
                g_loss.backward()
                optimizer_g.step()
            d_fake_g_mean, g_loss_mean, H_mean = np.mean(g_costs[-args.generator_iters:], 0)
            sum_loss_g += g_loss_mean.item() * len(data)

            if iteration % 250 == 0:
                writer.add_scalars('d_logit_mean', {'r_logit_mean': d_real_mean,
                                                    'f_logit_mean': d_fake_mean,
                                                    'G_f_logit_mean': d_fake_g_mean}, iteration)
                writer.add_scalar('D_loss', D_loss_mean, iteration)
                writer.add_scalar('gp_loss', gp_loss_mean, iteration)
                writer.add_scalar('g_loss', g_loss_mean, iteration)
                writer.add_scalar('H', H_mean, iteration)
            if iteration % 10000 == 0:
                with torch.no_grad():
                    self.visualize_results(data, epoch)
            if (iteration + 1) % 5000 == 0:
                avg_loss_d = sum_loss_d / args.batch_size / 5000
                avg_loss_g = sum_loss_g / args.batch_size / 5000
                print('(MEAN) ====> Epoch: {} Average loss d: {:.4f}'.format(epoch, avg_loss_d))
                print('(MEAN) ====> Epoch: {} Average loss g: {:.4f}'.format(epoch, avg_loss_g))
                sum_loss_d = 0
                sum_loss_g = 0
            if (iteration + 1) % 30000 == 0:
                self.save(epoch)


    def compute_entropy_acc(self, z):
        self.gen.eval()
        _, J = self.gen(z, True)
        jtj = torch.bmm(torch.transpose(J, -2, -1), J)
        H = 0.5 * torch.slogdet(jtj)[1].unsqueeze(-1)
        self.gen.train()
        return H.mean()

    def compute_entropy_mins_eval(self, z, ds=1):
        self.gen.eval()
        _, J = self.gen(z, True)
        s = torch.svd(J).S[:, -1:]
        H = (self.d / ds) * torch.log(s)
        self.gen.train()
        return H.mean()

    def plt_toy_density(self, logdensity, ax, npts=100,
                        title="$q(x)$", device="cpu", low=-4, high=4, exp=True):
        """
        Plot density of toy data.
        """
        side = np.linspace(low, high, npts)
        xx, yy = np.meshgrid(side, side)
        x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

        x = torch.from_numpy(x).type(torch.float32).to(device)
        logpx = logdensity(x).squeeze()

        if exp:
            logpx = logpx
            logpx = logpx - logpx.logsumexp(0)
            px = np.exp(logpx.cpu().detach().numpy()).reshape(npts, npts)
            px = px / px.sum()
        else:
            logpx = logpx - logpx.logsumexp(0)
            px = logpx.cpu().detach().numpy().reshape(npts, npts)

        ax.imshow(px, origin='lower', extent=[low, high, low, high])
        ax.set_title(title)

    def visualize_results(self, points, epoch, fix=True):
        self.gen.eval()
        self.disc.eval()

        points = points.detach().cpu().numpy()
        plt.clf()

        ax = plt.subplot(1, 3, 1, aspect="equal", title='data')
        ax.scatter(points[:, 0], points[:, 1], s=1)
        self.disc.cpu()

        ax = plt.subplot(1, 3, 2, aspect="equal")
        self.plt_toy_density(lambda x: -self.disc.main(x), ax,
                             low=-4, high=4,
                             title="p(x)")

        ax = plt.subplot(1, 3, 3, aspect="equal")
        self.plt_toy_density(lambda x: -self.disc.main(x), ax,
                             low=-4, high=4,
                             exp=False, title="log p(x)")

        plt.savefig(("{}/Dx_%08d.png" % epoch).format(self.result_dir))

        self.disc.to(device)
        self.gen.train()
        self.disc.train()

    def save(self, epoch):
        torch.save(self.disc.state_dict(), os.path.join(self.save_dir, 'epoch%03d' % epoch + '_d.pkl'))
        torch.save(self.gen.state_dict(), os.path.join(self.save_dir, 'epoch%03d' % epoch + '_g.pkl'))

    def load(self):

        pkl_path_d = '/home/congen/code/geoml_gan/models/two_moon/SNGAN/01/1619948832/epoch9999_d.pkl'
        pkl_path_g = '/home/congen/code/geoml_gan/models/two_moon/SNGAN/01/1619948832/epoch9999_g.pkl'

        self.disc.load_state_dict(torch.load(pkl_path_d))
        self.gen.load_state_dict(torch.load(pkl_path_g))



    def compute_ebmpr(self):

        pkl_path_d = '/home/congen/code/EBM-BB/models/25gaussians/EBM_0GP/01/1639843173/epoch149999_d.pkl'
        pkl_path_g = '/home/congen/code/EBM-BB/models/25gaussians/EBM_0GP/01/1639843173/epoch149999_g.pkl'
        self.disc.load_state_dict(torch.load(pkl_path_d))
        self.gen.load_state_dict(torch.load(pkl_path_g))
        self.disc.eval()
        self.gen.eval()
        points = data_process(args.dataset, args.batch_size)
        points = points.to(self.device)
        points = points.detach().cpu().numpy()
        plt.clf()

        ax = plt.subplot(1, 3, 1, aspect="equal", title='data')
        ax.scatter(points[:, 0], points[:, 1], s=1)
        self.disc.cpu()

        ax = plt.subplot(1, 3, 2, aspect="equal")
        self.plt_toy_density(lambda x: -self.disc.main(x), ax,
                             low=-4, high=4,
                             title="p(x)")

        ax = plt.subplot(1, 3, 3, aspect="equal")
        self.plt_toy_density(lambda x: -self.disc.main(x), ax,
                             low=-4, high=4,
                             exp=False, title="log p(x)")

        plt.show()
        plt.close()

class Disc(nn.Module):
    def __init__(self, input_dim=2, dim=100):
        super().__init__()
        if args.sn:
            self.main = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(input_dim, dim, bias=True)),
                nn.PReLU(),
                nn.utils.spectral_norm(nn.Linear(dim, dim, bias=True)),
                nn.PReLU(),
                nn.utils.spectral_norm(nn.Linear(dim, 1, bias=True))
            )
        else:
            self.main = nn.Sequential(
                nn.Linear(input_dim, dim, bias=True),
                nn.PReLU(),
                nn.Linear(dim, dim, bias=True),
                nn.PReLU(),
                nn.Linear(dim, 1, bias=True))
        #self.apply(weights_init)
        self.l = nn.Linear(1, 1, bias=False)
    def forward(self, x, return_fmap=False):
        energy = self.main(x)
        out = self.l(energy)
        return out
class gen(nn.Module):
    def __init__(self, input_dim=2, dim=100,z_dim=2):
        super().__init__()
        self.main = nnj.Sequential(
            nnj.Linear(z_dim, dim, bias=True),
            nnj.PReLU(),
            nnj.BatchNorm1d(dim),
            nnj.Linear(dim, dim, bias=True),
            nnj.PReLU(),
            nnj.BatchNorm1d(dim),
            nnj.Linear(dim, input_dim, bias=True))
    def forward(self,z,jacob=False):
        out = self.main(z,jacob)
        return out
@utils.register_model(name='EBM_BB')
class EBM_BB(nn.Module):
    def __init__(self, args, layers, device='cpu'):
        super(EBM_BB, self).__init__()

        self.device = device
        self.D = int(layers[0])  # Dimension of x
        self.d = int(layers[-1])  # Dimension of z
        self.h = layers[1:-1]  # Dimension of hidden layers
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.model_name = args.EBM_type
        self.sample_z_ = torch.randn((self.batch_size, self.d))
        self.sample_z_ = self.sample_z_.to(self.device)

        self.disc = Disc()
        self.gen = gen()
        self.to(self.device)


    def trainer(self, iters=50):

        optimizer_d = torch.optim.Adam(self.disc.parameters(), lr=args.lrd,
                                           betas=(0.0, 0.9), eps=1e-6)
        optimizer_g = torch.optim.Adam(self.gen.parameters(), lr=args.lrg,
                                           betas=(0.0, 0.9), eps=1e-6)
        sum_loss_d = 0
        sum_loss_g = 0
        writer = SummaryWriter(log_dir=self.log_dir)
        for iteration in range(iters):
            epoch = iteration
            e_costs = []
            g_costs = []
            #energy function
            for i in range(args.energy_model_iters):
                data=data_process(args.dataset,args.batch_size)
                data = data.to(self.device)
                z_train = torch.randn((data.shape[0], self.d)).to(self.device)
                z_train.requires_grad_()
                G_z = self.gen(z_train)
                d_real = self.disc(data)
                d_fake = self.disc(G_z)
                gradients = torch.autograd.grad(
                    outputs=d_fake,
                    inputs=G_z,
                    grad_outputs=torch.ones_like(d_fake),
                    allow_unused=True,
                    create_graph=True,
                )[0]
                gradients = gradients.flatten(start_dim=1)
                Jv, gp = self.compute_gp(z_train)
                gp_loss = (((gradients * Jv.detach()).sum(-1) - gp.detach()) ** 2).mean() * 0.5
                if args.ada == True:
                    gp_weight = 0.7 * (2307 + iteration) ** (-0.55)
                    D_loss = (d_real - d_fake).mean() +  (F.relu(gp_weight*gp_loss- args.thre))
                else:
                    D_loss = (d_real - d_fake).mean() + (F.relu(args.gp_weight*gp_loss- args.thre))

                e_costs.append([d_real.mean().item(), d_fake.mean().item(), D_loss.item(), gp_loss.item()])
                optimizer_d.zero_grad()
                D_loss.backward()
                optimizer_d.step()

            d_real_mean, d_fake_mean, D_loss_mean, gp_loss_mean = np.mean(e_costs[-args.energy_model_iters:], 0)
            sum_loss_d += D_loss_mean.item() * len(data)
           #generator
            for i in range(args.generator_iters):
                z_train = torch.randn((data.shape[0], self.d))
                z_train = z_train.to(self.device)
                z_train.requires_grad_()
                if args.train_mode == 'acc':
                    fake = self.gen(z_train)
                    d_fake_g = self.disc(fake)
                    H = self.compute_entropy_acc(z_train)

                else:
                    fake = self.gen(z_train)
                    d_fake_g = self.disc(fake)
                    H = self.compute_entropy_mins_eval(z_train)


                g_loss = d_fake_g.mean() - H * args.H_weight

                optimizer_g.zero_grad()
                g_loss.backward()
                optimizer_g.step()

                g_costs.append([d_fake_g.mean().item(), g_loss.item(), H.mean().item()])

            d_fake_g_mean, g_loss_mean, H_mean = np.mean(g_costs[-args.generator_iters:], 0)
            sum_loss_g += g_loss_mean.item() * len(data)

            if iteration % 250 == 0:
                writer.add_scalars('d_logit_mean', {'r_logit_mean': d_real_mean,
                                                    'f_logit_mean': d_fake_mean,
                                                    'G_f_logit_mean': d_fake_g_mean}, iteration)
                writer.add_scalar('D_loss', D_loss_mean, iteration)
                writer.add_scalar('gp_loss', gp_loss_mean, iteration)
                writer.add_scalar('g_loss', g_loss_mean, iteration)
                writer.add_scalar('H', H_mean, iteration)

            if iteration % 10000 == 0:
                with torch.no_grad():
                    self.visualize_results(data, epoch)
            if (iteration + 1) % 5000 == 0:
                avg_loss_d = sum_loss_d / args.batch_size / 5000
                avg_loss_g = sum_loss_g / args.batch_size / 5000
                print('(MEAN) ====> Epoch: {} Average loss d: {:.4f}'.format(epoch, avg_loss_d))
                print('(MEAN) ====> Epoch: {} Average loss g: {:.4f}'.format(epoch, avg_loss_g))
                sum_loss_d = 0
                sum_loss_g = 0
            if (iteration + 1) % 30000 == 0:
                self.save(epoch)


    def compute_entropy_acc(self, z):
        self.gen.eval()
        _, J = self.gen(z, True)
        jtj = torch.bmm(torch.transpose(J, -2, -1), J)
        H = 0.5 * torch.slogdet(jtj)[1].unsqueeze(-1)
        self.gen.train()
        return H.mean()

    def compute_gp(self, z):
        self.gen.eval()
        z.requires_grad_()
        logpz = td.Normal(loc=torch.zeros(self.d).to(device), scale=torch.ones(self.d).to(device)).log_prob(
            z).sum(1)
        fake, J = self.gen(z, True)
        v = torch.randn(z.shape).to(self.device)
        if args.train_mode == 'acc':
            jtj = torch.bmm(torch.transpose(J, -2, -1), J)
            detjtj = 0.5 * torch.slogdet(jtj)[1]
            logpGz = logpz - detjtj
        elif args.train_mode == 'mins':
            s = torch.svd(J).S[:, -1:]
            H = self.d * torch.log(s)
            logpGz = logpz - H.squeeze()
        deri = torch.autograd.grad(-logpGz, z, torch.ones_like(logpGz, requires_grad=True), allow_unused=True,
                                   create_graph=True)[0]
        gp = (deri * v).sum(-1)
        Jv = J.bmm(v.unsqueeze(-1)).squeeze(-1)
        self.gen.train()
        return Jv, gp


    def compute_entropy_mins_eval(self, z):
        self.gen.eval()
        _, J = self.gen(z, True)
        s = torch.svd(J).S[:, -1:]
        H = self.d  * torch.log(s)
        self.gen.train()
        return H.mean()


    def plt_toy_density(self, logdensity, ax, npts=100,
                        title="$q(x)$", device="cpu", low=-4, high=4, exp=True):
        """
        Plot density of toy data.
        """
        side = np.linspace(low, high, npts)
        xx, yy = np.meshgrid(side, side)
        x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

        x = torch.from_numpy(x).type(torch.float32).to(device)

        logpx = logdensity(x).squeeze()

        if exp:
            logpx = logpx
            logpx = logpx - logpx.logsumexp(0)

            px = np.exp(logpx.cpu().detach().numpy()).reshape(npts, npts)
            px = px / px.sum()

        else:
            logpx = logpx - logpx.logsumexp(0)
            px = logpx.cpu().detach().numpy().reshape(npts, npts)

        ax.imshow(px, origin='lower', extent=[low, high, low, high])
        ax.set_title(title)

    def visualize_results(self, points, epoch, fix=True):
        self.gen.eval()
        self.disc.eval()

        points = points.detach().cpu().numpy()
        plt.clf()

        ax = plt.subplot(1, 3, 1, aspect="equal", title='data')
        ax.scatter(points[:, 0], points[:, 1], s=1)
        self.disc.cpu()

        ax = plt.subplot(1, 3, 2, aspect="equal")
        self.plt_toy_density(lambda x: -self.disc.main(x), ax,
                             low=-4, high=4,
                             title="p(x)")

        ax = plt.subplot(1, 3, 3, aspect="equal")
        self.plt_toy_density(lambda x: -self.disc.main(x), ax,
                             low=-4, high=4,
                             exp=False, title="log p(x)")

        plt.savefig(("{}/Dx_%08d.png" % epoch).format(self.result_dir))

        self.disc.to(device)
        self.gen.train()
        self.disc.train()

    def save(self, epoch):

        torch.save(self.disc.state_dict(), os.path.join(self.save_dir, 'epoch%03d' % epoch + '_d.pkl'))
        torch.save(self.gen.state_dict(), os.path.join(self.save_dir, 'epoch%03d' % epoch + '_g.pkl'))


    def load(self):

        pkl_path_d = '/home/cong/code/geoml_gan/models/two_moon/EBM_upper_gp/01/1620923311/epoch19999_d.pkl'
        pkl_path_g = '/home/cong/code/geoml_gan/models/two_moon/EBM_upper_gp/01/1620923311/epoch19999_g.pkl'

        self.disc.load_state_dict(torch.load(pkl_path_d))
        self.gen.load_state_dict(torch.load(pkl_path_g))


    def compute_ebmpr(self):

        pkl_path_d = '/home/congen/code/EBM-BB/models/swiss_roll/EBM_BB/01/1639848355/epoch119999_d.pkl'
        pkl_path_g = '/home/congen/code/EBM-BB/models/swiss_roll/EBM_BB/01/1639848355/epoch119999_g.pkl'
        self.disc.load_state_dict(torch.load(pkl_path_d))
        self.gen.load_state_dict(torch.load(pkl_path_g))
        self.disc.eval()
        self.gen.eval()
        points = data_process(args.dataset, args.batch_size)
        points = points.to(self.device)
        points = points.detach().cpu().numpy()
        plt.clf()
        ax = plt.subplot(1, 3, 1, aspect="equal", title='data')
        ax.scatter(points[:, 0], points[:, 1], s=1)

        self.disc.cpu()
        ax = plt.subplot(1, 3, 2, aspect="equal")
        self.plt_toy_density(lambda x: -self.disc.main(x), ax,
                             low=-4, high=4,
                             title="p(x)")

        ax = plt.subplot(1, 3, 3, aspect="equal")
        self.plt_toy_density(lambda x: -self.disc.main(x), ax,
                             low=-4, high=4,
                             exp=False, title="log p(x)")

        plt.show()
        plt.close()


if __name__ == "__main__":
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=3
        export PORT=6006
        export CUDA_HOME=/opt/cuda/cuda-10.2
        export TIME_STR=1
        export PYTHONPATH=./
        python ./train/train_toy.py


    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    os.environ['CUDA_HOME'] = '/opt/cuda/cuda-10.2'
    desc = "Pytorch implementation of EBM collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--EBM_type', type=str, default='EBM_0GP',
                        choices=['EBM_BB','EBM_0GP'],
                        help='The type of EBM')
    parser.add_argument('--dataset', type=str, default='swiss_roll',
                        choices=['swiss_roll',  'two_moon', '25gaussians'],
                        help='The name of dataset')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train',  'ebmpr'], help='mode')
    parser.add_argument('--iters', type=int, default=150001, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=200, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=2, help='The size of input image')
    parser.add_argument('--energy_model_iters', type=int, default=1)
    parser.add_argument("--generator_iters", type=int, default=1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument('--train_mode', type=str, default='mins',
                        choices=['acc',  'mins'],help='mode')
    parser.add_argument("--bn", type=bool, default=True)
    parser.add_argument("--sn", type=bool, default=True)
    parser.add_argument("--ada", type=bool, default=True)
    parser.add_argument("--thre", type=int, default=0)
    parser.add_argument("--lrd", type=float, default=2e-4)
    parser.add_argument("--lrg", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--gp", type=str, default=False)
    parser.add_argument("--gp_weight", type=float, default=0.001)
    parser.add_argument("--H_weight", type=float, default=1)
    parser.add_argument("--detach", type=str, default=False)
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the network')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--benchmark_mode', type=bool, default=True)

    args = parser.parse_args()
    # --save_dir
    if args.mode == 'train' and utils.is_debugging() == False:
        time = int(time.time())
        args.save_dir = os.path.join(args.save_dir + '/' + args.dataset + '/'
                                     + args.EBM_type + '/%02d' % args.energy_model_iters + '/%03d' % time)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # --result_dir
        args.result_dir = os.path.join(args.result_dir + '/' + args.dataset + '/'
                                       + args.EBM_type + '/%02d' % args.energy_model_iters + '/%03d' % time)
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)

        # --log_dir
        args.log_dir = os.path.join(args.log_dir + '/' + args.dataset + '/'
                                    + args.EBM_type + '/%02d' % args.energy_model_iters + '/tb_%03d' % time)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        with open("{}/args.txt".format(args.result_dir), 'w') as f:
            json.dump(args.__dict__, f, indent=4, sort_keys=True)
    elif utils.is_debugging() == True:
        args.save_dir = os.path.join(args.save_dir + '/debug')
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # --result_dir
        args.result_dir = os.path.join(args.result_dir + '/debug')
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)

        # --result_dir
        args.log_dir = os.path.join(args.log_dir + '/debug')
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        with open("{}/args.txt".format(args.result_dir), 'w') as f:
            json.dump(args.__dict__, f, indent=4, sort_keys=True)
    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    utils.setup_seed(args.seed)
    layers = [2, 100, 100, 2]
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model= utils.get_model(args.EBM_type)(args, layers, device)

    if args.mode == 'train':
        model.trainer(args.iters)

    elif args.mode == 'ebmpr':
        model.compute_ebmpr()




