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

class EBM_0gp(nn.Module):
    def __init__(self, args, layers, device='cpu'):
        super(EBM_0gp, self).__init__()

        self.device = device
        self.D = int(layers[0])  # Dimension of x
        self.d = int(layers[-1])  # Dimension of z
        self.h = layers[1:-1]  # Dimension of hidden layers
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.sample_z_ = torch.randn((self.batch_size, self.d))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()
        # Set up mean discriminator/generator
        disc = []
        for k in range(len(layers) - 2):
            in_features = int(layers[k])
            out_features = int(layers[k + 1])
            # disc.append(nnj.ResidualBlock(nnj.Linear(in_features, out_features),
            #                              nnj.Softplus())
            if args.sn == True:
                disc.append(nn.utils.spectral_norm(nn.Linear(in_features, out_features, bias=True)))
            else:
                disc.append(nn.Linear(in_features, out_features, bias=True))
            # disc.append(nn.Linear(in_features, out_features, bias=True))
            # disc.append(nn.utils.spectral_norm(torch.nn.Conv1d(in_channels=in_features, out_channels=out_features,
            # kernel_size=1, stride=1, padding=0)))
            disc.append(nn.PReLU())
            # disc.append(nnj.LeakyReLU(.2, inplace=True))
        if args.sn == True:
            disc.append(nn.utils.spectral_norm(nn.Linear(out_features, 1, bias=True)))
        else:
            disc.append(nn.Linear(out_features, 1, bias=True))

        gen = []
        for k in reversed(range(1, len(layers) - 1)):
            in_features = int(layers[k + 1])
            out_features = int(layers[k])
            # gen.append(nnj.ResidualBlock(nnj.Linear(in_features, out_features),
            # nnj.LeakyReLU(0.2)))
            # gen.append(torch.nn.Conv1d(in_channels=in_features, out_channels=out_features,
            # kernel_size=1, stride=1, padding=0))
            gen.append(nnj.Linear(in_features, out_features, bias=True))
            gen.append(nnj.PReLU())
            if args.bn == True:
                gen.append(nnj.BatchNorm1d(out_features, momentum=args.momentum, affine=True))
            # gen.append(nnj.Linear(in_features, out_features))
            # gen.append(nnj.LeakyReLU())
        # gen.append(torch.nn.Conv1d(in_channels=out_features, out_channels=self.D,
        # kernel_size=1, stride=1, padding=0))
        gen.append(nnj.Linear(out_features, self.D, bias=True))
        # gen.append(nnj.Tanh())

        # self.disc = nn.Sequential(*disc)
        self.disc = Disc()
        self.gen = nnj.Sequential(*gen)
        if self.gpu_mode:
            self.gen.cuda()
            self.disc.cuda()
        self.to(self.device)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def trainer(self, data_loader, iters=50):
        # self.load()
        # criterion = torch.nn.BCEWithLogitsLoss()
        if args.optimizer == "SGD":
            optimizer_d = torch.optim.SGD(self.disc.parameters(), lr=args.lrd, momentum=0.8)
            optimizer_g = torch.optim.SGD(self.gen.parameters(), lr=args.lrg, momentum=0.8)

        elif args.optimizer == "RMSprop":
            optimizer_d = torch.optim.RMSprop(self.disc.parameters(), lr=args.lrd, alpha=0.9)
            optimizer_g = torch.optim.RMSprop(self.gen.parameters(), lr=args.lrg, alpha=0.9)
        elif args.optimizer == "Adam":
            optimizer_d = torch.optim.Adam(self.disc.parameters(), lr=args.lrd,
                                           betas=(0.0, 0.9), eps=1e-6)
            optimizer_g = torch.optim.Adam(self.gen.parameters(), lr=args.lrg,
                                           betas=(0.0, 0.9), eps=1e-6)

        # optimizer_d = torch.optim.Adam(self.disc.parameters(), betas=(0.0, 0.9), lr=args.lrd)
        # optimizer_g = torch.optim.Adam(self.gen.parameters(), betas=(0.0, 0.9), lr=args.lrg)
        d_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=args.milestone, gamma=args.gammad)
        g_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_g, milestones=args.milestone, gamma=args.gammag)
        # g2_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_g2, milestones=args.milestone, gamma=args.gammag)
        sum_loss_d = 0
        sum_loss_g = 0
        writer = SummaryWriter(log_dir=self.log_dir)
        for iteration in range(iters):
            # if iteration < 30000:
            #     w = iteration/iters
            epoch = (iteration) // len(data_loader)
            e_costs = []
            g_costs = []
            # for batch_idx, (data,) in enumerate(data_loader):
            for i in range(args.energy_model_iters):
                if args.dataset == 'two_moon':
                    data, y = sklearn.datasets.make_moons(n_samples=self.batch_size, noise=0.1)
                elif args.dataset == 'swiss_roll':
                    data = sklearn.datasets.make_swiss_roll(n_samples=self.batch_size, noise=1.0)[0]
                    data = data.astype("float32")[:, [0, 2]]
                    data /= 5
                elif args.dataset == '25gaussians':
                    data = []
                    for j in range(1000 // 25):
                        for x in range(-2, 3):
                            for y in range(-2, 3):
                                point = np.random.randn(2) * 0.05
                                point[0] += 2 * x
                                point[1] += 2 * y
                                data.append(point)
                    data = np.array(data[:self.batch_size], dtype='float32')
                    np.random.shuffle(data)
                    data /= 2.828
                data = torch.Tensor(data)
                # summary_d = collections.defaultdict(dict)
                data = data.to(self.device)
                data.requires_grad_()
                # discriminator
                # for i in range(n_disc):
                z_train = torch.randn((data.shape[0], self.d))
                if self.gpu_mode:
                    z_train = z_train.cuda()
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
                # L2 norm
                gp_loss = (gradients.norm(2, dim=1) ** 2).mean()
                D_loss = (d_real - d_fake).mean() + gp_loss * args.gp_weight
                # D_loss=d_real.mean()+ub+gp_loss * args.gp_weight

                e_costs.append([d_real.mean().item(), d_fake.mean().item(), D_loss.item(), gp_loss.item()])
                optimizer_d.zero_grad()
                D_loss.backward()
                optimizer_d.step()
            d_real_mean, d_fake_mean, D_loss_mean, gp_loss_mean = np.mean(e_costs[-args.energy_model_iters:], 0)
            sum_loss_d += D_loss_mean.item() * len(data)

            # generator
            for i in range(args.generator_iters):
                z_train = torch.randn((data.shape[0], self.d))

                if self.gpu_mode:
                    z_train = z_train.cuda()
                    z_train.requires_grad_()
                if args.train_mode == 'train':
                    fake, J = self.gen(z_train, True)
                    d_fake_g = self.disc(fake)
                    H = self.compute_entropy_mins(J)
                # fake = self.gen(z_train)
                # J=self.compute_jacobian(z_train)
                elif args.train_mode == 'acc':
                    max_dis = (data - data.mean(0)).norm(2, dim=-1).max()
                    std_dis = data.var(dim=0).sum().sqrt()
                    # z_select=z_train[(z_train.norm(2,dim=1)<1)]
                    fake = self.gen(z_train)
                    z_select = z_train[(fake - data.mean(dim=0)).norm(2, -1) < 3 * std_dis]
                    d_fake_g = self.disc(fake)
                    # thre=d_fake_g.quantile(0.75)
                    # H = self.compute_entropy_acc(z_train[(d_fake_g < thre).squeeze()])
                    H = self.compute_entropy_acc(z_select)
                else:
                    fake = self.gen(z_train)
                    d_fake_g = self.disc(fake)
                    H = self.compute_entropy_mins_eval(z_train)

                # H=d_fake_g
                # g_loss = criterion(d_fake_g, torch.Tensor(d_fake_g.shape).fill_(1).cuda())
                # ub= self.compute_upperbound(z_train, logpz, d_fake_g)
                # if ub<5:
                # w=(d_real_g - d_fake_g).mean().abs().detach() * ub.detach()
                # w = (-(d_fake_g).std()).exp().detach()
                # w=(-d_fake_g-((-d_fake_g).logsumexp(0))).exp()
                g_loss = d_fake_g.mean() - H
                # else:
                # g_loss = (d_fake_g-wh*H).mean()
                g_costs.append([d_fake_g.mean().item(), g_loss.item(), H.mean().item()])
                # optimizer_d.zero_grad()
                optimizer_g.zero_grad()
                g_loss.backward()
                # optimizer_d.step()
                optimizer_g.step()
            d_fake_g_mean, g_loss_mean, H_mean = np.mean(g_costs[-args.generator_iters:], 0)
            sum_loss_g += g_loss_mean.item() * len(data)
            if (iteration + 1) % 1000 == 0:
                d_lr_scheduler.step()
                g_lr_scheduler.step()
                # g2_lr_scheduler.step()
            if iteration % 250 == 0:
                writer.add_scalars('d_logit_mean', {'r_logit_mean': d_real_mean,
                                                    'f_logit_mean': d_fake_mean,
                                                    'G_f_logit_mean': d_fake_g_mean}, iteration)
                writer.add_scalar('D_loss', D_loss_mean, iteration)
                writer.add_scalar('gp_loss', gp_loss_mean, iteration)
                writer.add_scalar('g_loss', g_loss_mean, iteration)
                writer.add_scalar('H', H_mean, iteration)
            if iteration % 5000 == 0:
                with torch.no_grad():
                    self.visualize_results2(data, epoch)
            if (iteration + 1) % 5000 == 0:
                avg_loss_d = sum_loss_d / len(data_loader) / args.batch_size / 5000
                avg_loss_g = sum_loss_g / len(data_loader) / args.batch_size / 5000
                print('(MEAN) ====> Epoch: {} Average loss d: {:.4f}'.format(epoch, avg_loss_d))
                print('(MEAN) ====> Epoch: {} Average loss g: {:.4f}'.format(epoch, avg_loss_g))
                sum_loss_d = 0
                sum_loss_g = 0
            if (iteration + 1) % 10000 == 0:
                self.save(epoch)
            # if batch_idx % 50 == 0:
            # summary_defaultdict2txtfig(default_dict=summary_d, prefix='GAN', step=epoch*len(data_loader) + batch_idx,
            # textlogger=self.myargs.textlogger, save_fig_sec=60)

    def refineD(self, data_loader, num_epochs=50, ):

        self.load()
        self.gen.eval()
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer_d = torch.optim.Adam(self.disc.parameters(), weight_decay=1e-4, lr=2e-4)
        writer = SummaryWriter(
            log_dir=self.log_dir + '/' + self.dataset + '/' + self.model_name + '/refineD/' + '%03d' % self.d + '/tb')
        for epoch in range(num_epochs):
            # epoch=epoch+300
            sum_loss_d = 0
            for batch_idx, (data,) in enumerate(data_loader):
                summary_d = collections.defaultdict(dict)
                data = data.to(self.device)
                data.requires_grad_()
                # discriminator
                # for i in range(n_disc):
                z_train = torch.randn((self.batch_size, self.d))
                if self.gpu_mode:
                    z_train = z_train.cuda()
                G_z = self.gen(z_train)
                d_real = self.disc(data)
                d_fake = self.disc(G_z.detach())
                # Real data Discriminator loss
                errD_real = criterion(d_real, torch.Tensor(d_real.shape).fill_(1).cuda())
                # Fake data Discriminator loss
                errD_fake = criterion(d_fake, torch.Tensor(d_fake.shape).fill_(0).cuda())
                gradients = torch.autograd.grad(
                    outputs=d_real,
                    inputs=data,
                    grad_outputs=torch.ones_like(d_real),
                    allow_unused=True,
                    create_graph=True,
                )[0]
                gradients = gradients.flatten(start_dim=1)
                # L2 norm
                gp_loss = (gradients.norm(2, dim=1) ** 2).mean()
                D_loss = (errD_real + errD_fake) / 2 + gp_loss
                # D_loss = errD_real  + Q * 10
                summary_d['d_logit_mean']['r_logit_mean'] = d_real.mean().item()
                summary_d['d_logit_mean']['f_logit_mean'] = d_fake.mean().item()
                summary_d['gp_loss']['gp_loss'] = gp_loss.item()
                summary_d['D_loss']['D_loss'] = D_loss.item()
                sum_loss_d += D_loss.item() * len(data)
                optimizer_d.zero_grad()
                D_loss.backward()
                optimizer_d.step()
                if batch_idx % 250 == 0:
                    writer.add_scalars('d_logit_mean', {'r_logit_mean': summary_d['d_logit_mean']['r_logit_mean'],
                                                        'f_logit_mean': summary_d['d_logit_mean']['f_logit_mean']},
                                       batch_idx + epoch * len(data_loader))
                    writer.add_scalar('D_loss', summary_d['D_loss']['D_loss'], batch_idx + epoch * len(data_loader))
                    writer.add_scalar('gp_loss', summary_d['gp_loss']['gp_loss'], batch_idx + epoch * len(data_loader))
            if (epoch + 1) % 50 == 0:
                with torch.no_grad():
                    self.visualize_results_refineD((epoch))

            avg_loss_d = sum_loss_d / len(data_loader.dataset)
            print('(MEAN) ====> Epoch: {} Average loss d: {:.4f}'.format(epoch, avg_loss_d))
            if (epoch + 1) % 100 == 0:
                self.save_refineD(epoch)

    def create_figure(self, rows=1, cols=1, figsize=5):
        fig = plt.figure(figsize=(figsize * cols, figsize * rows))
        fig.tight_layout()
        return fig

    def compute_entropy(self, z, ds=2):
        self.gen.eval()
        fake, J2 = self.gen(z, True)
        # delta = torch.svd(z).V[:, :ds].t().unsqueeze(0).repeat(z.shape[0],1,1)
        delta = torch.eye(2).repeat(z.shape[0], 1, 1).cuda()
        delta.requires_grad_()
        J = torch.zeros(z.shape[0], ds, self.D)
        for i in range(ds):
            projection = torch.ones_like(fake, requires_grad=True)
            intermediate = torch.autograd.grad(fake, z, projection, create_graph=True)
            Jv = torch.autograd.grad(intermediate[0], projection, delta[:, i, :], create_graph=True)[0]
            J[:, :, i] = Jv
        # est = (self.d / ds) * torch.slogdet(J.bmm(J.permute(0,2,1)))[1]
        s = torch.svd(J).S[:, -1:]
        # Jv=J.bmm(delta)
        H = (self.d / ds) * torch.log(s)
        # H=0.5*est.unsqueeze(-1)
        # self.gen.train()
        return H

    def compute_entropy_acc(self, z):
        self.gen.eval()
        _, J = self.gen(z, True)
        jtj = torch.bmm(torch.transpose(J, -2, -1), J)
        H = 0.5 * torch.slogdet(jtj)[1].unsqueeze(-1)
        self.gen.train()
        return H.mean()

    def compute_entropy_mins(self, J, ds=1):
        # self.gen.eval()
        # _,J=self.gen(z,True)
        s = torch.svd(J).S[:, -1:]
        H = (self.d / ds) * torch.log(s)
        # self.gen.train()
        return H

    def compute_upperbound(self, z, logpz, disc_fake, ds=1):
        self.gen.eval()
        _, J = self.gen(z, True)
        s = torch.svd(J).S[:, -1:]
        smax = torch.svd(J).S[:, :1]
        v = torch.randn(z.shape).unsqueeze(-1).to(self.device)
        # JF=(self.d/2)*torch.log(1/self.d*(J.bmm(v).norm(2,dim=[-2,-1])**2)/(v.norm(2,dim=[-2,-1])**2))
        JF = (self.d / 2) * torch.log(1 / self.d * (J.bmm(v).norm(2, dim=[-2, -1]) ** 2))
        # JF =(self.d / ds)* torch.log(smax).squeeze(-1)
        H = (self.d / ds) * torch.log(s).squeeze(-1)
        jtj = torch.bmm(torch.transpose(J, -2, -1), J)
        detjtj = 0.5 * torch.slogdet(jtj)[1]
        # disc_fake=self.disc(G_z)
        logm = (-self.v * disc_fake.squeeze(-1) - logpz + H).min()
        logM = (-self.v * disc_fake.squeeze(-1) - logpz + JF).max()
        # z_theta=(-disc_fake.squeeze(-1)-logpz+JF).exp().mean().log()
        # m=logm.exp()
        # M=logM.exp()
        # ub=0.25*(M-m)*(1/(m)-1/(M))
        ub = torch.abs(logM - logm)

        self.gen.train()
        return ub

    def compute_entropy_mins_eval(self, z, ds=1):
        self.gen.eval()
        _, J = self.gen(z, True)
        s = torch.svd(J).S[:, -1:]
        H = (self.d / ds) * torch.log(s)
        self.gen.train()
        return H.mean()

    def log_figure(self, fig, dir=None, view=False, saved_file=None, iteration=0):
        if view:
            fig.show()

        if saved_file:
            saved_path = os.path.join(dir, f'{saved_file}_{iteration:08d}.png')
            # fig.savefig(saved_path, bbox_inches='tight', pad_inches=0.01)
            fig.savefig(saved_path, pad_inches=0.01)

        plt.close(fig)
        pass

    def add_axes_for_2D_points(self, fig, axes_index, points, colors):

        ax = fig.add_subplot(int(axes_index))

        points = points.detach().squeeze()
        points = points.cpu().numpy()
        ax.scatter(points[:, 0], points[:, 1], c=colors)
        pass

    def visualize_results(self, points, epoch, fix=True):
        self.gen.eval()
        self.disc.eval()
        # results_dir = os.path.join(self.result_dir + '/' + self.dataset + '/'
        #  + self.model_name + '/' + '%03d' % self.d + '_1bnH')

        # if not os.path.exists(results_dir):
        # os.makedirs(results_dir)

        # points, y = sklearn.datasets.make_moons(n_samples=500, noise=0.1)
        # points = torch.Tensor(points).cuda()
        if fix:
            """ fixed noise """
            samples = self.gen(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.randn((points.shape[0], self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.gen(sample_z_)

        fig = self.create_figure(rows=1, cols=3)
        colors = samples[:, 1].detach().cpu().numpy()
        self.add_axes_for_2D_points(fig=fig, axes_index=131, points=points, colors=colors)
        self.add_axes_for_2D_points(fig=fig, axes_index=132, points=samples, colors=colors)
        self.add_axes_for_2D_points(fig=fig, axes_index=133, points=self.sample_z_, colors=colors)
        self.log_figure(fig=fig, dir=self.result_dir, saved_file='x_z_recon_x', iteration=epoch)
        fig2 = self.create_figure(rows=1, cols=2)
        ax1 = fig2.add_subplot(121)
        ax2 = fig2.add_subplot(122)
        x_lin = np.linspace(-3, 3, 100)
        y_lin = np.linspace(-3, 3, 100)
        xx, yy = np.meshgrid(x_lin, y_lin)
        X_grid = np.column_stack([xx.flatten(), yy.flatten()])
        with torch.no_grad():
            Dx = (-self.disc(torch.from_numpy(X_grid).float().cuda()))
            Dx = Dx - Dx.logsumexp(0)
            EDx = Dx.exp()
            EDx = EDx / EDx.sum()
        DX = Dx.reshape(xx.shape)
        EDX = EDx.reshape(xx.shape)
        # ax1.contourf(x_lin, y_lin, DX.detach().cpu(), cmap='cividis')
        ax1.imshow(EDX.detach().cpu(), extent=(-3, 3, -3, 3), origin='lower')
        # plt.colorbar()
        # self.log_figure(fig=fig2, dir=results_dir,saved_file='EDx', iteration=epoch)
        # self.gen.train()
        # self.disc.train()
        ax2.imshow(DX.detach().cpu(), extent=(-3, 3, -3, 3), origin='lower')
        # plt.colorbar()
        self.log_figure(fig=fig2, dir=self.result_dir, saved_file='Dx', iteration=epoch)
        self.gen.train()
        self.disc.train()
        # grid2 = torchvision.utils.make_grid(samples.data, scale_each=True, range=(0, 1), normalize=True)
        # train_writer.add_image('generate', grid, epoch * 1875 + batch_idx)

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
            # logpx = logpx - logpx.mean(0)
            px = np.exp(logpx.cpu().detach().numpy()).reshape(npts, npts)
            px = px / px.sum()
            # px = F.log_softmax(logpx, 0).exp().cpu().detach().numpy().reshape(npts, npts)
        else:
            logpx = logpx - logpx.logsumexp(0)
            px = logpx.cpu().detach().numpy().reshape(npts, npts)

        ax.imshow(px)
        ax.set_title(title)

    def visualize_results2(self, points, epoch, fix=True):
        self.gen.eval()
        self.disc.eval()
        # results_dir = os.path.join(self.result_dir + '/' + self.dataset + '/'
        #  + self.model_name + '/' + '%03d' % self.d + '_1bnH')

        # if not os.path.exists(results_dir):
        # os.makedirs(results_dir)

        # points, y = sklearn.datasets.make_moons(n_samples=500, noise=0.1)
        # points = torch.Tensor(points).cuda()
        if fix:
            """ fixed noise """
            samples = self.gen(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.randn((points.shape[0], self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.gen(sample_z_)
        samples = samples.detach().cpu().numpy()
        points = points.detach().cpu().numpy()
        plt.clf()
        ax = plt.subplot(1, 4, 1, aspect="equal", title='gen')
        ax.scatter(samples[:, 0], samples[:, 1], s=1)

        ax = plt.subplot(1, 4, 2, aspect="equal", title='data')
        ax.scatter(points[:, 0], points[:, 1], s=1)
        self.disc.cpu()

        ax = plt.subplot(1, 4, 3, aspect="equal")
        self.plt_toy_density(lambda x: -self.disc.main(x), ax,
                             low=-4, high=4,
                             title="p(x)")

        ax = plt.subplot(1, 4, 4, aspect="equal")
        self.plt_toy_density(lambda x: -self.disc.main(x), ax,
                             low=-4, high=4,
                             exp=False, title="log p(x)")

        plt.savefig(("{}/Dx_%08d.png" % epoch).format(self.result_dir))

        self.disc.to(device)
        self.gen.train()
        self.disc.train()

    def save(self, epoch):
        # save_dir = os.path.join(self.save_dir, self.dataset, self.model_name,'%03d'%self.d+'_1bnH')

        # if not os.path.exists(save_dir):
        # os.makedirs(save_dir)

        torch.save(self.disc.state_dict(), os.path.join(self.save_dir, 'epoch%03d' % epoch + '_d.pkl'))
        torch.save(self.gen.state_dict(), os.path.join(self.save_dir, 'epoch%03d' % epoch + '_g.pkl'))
        # with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
        #     pickle.dump(self.train_hist, f)

    def load(self):
        # save_dir = os.path.join(self.save_dir, self.dataset, self.model_name,'%03d'%self.d)
        pkl_path_d = '/home/congen/code/geoml_gan/models/two_moon/SNGAN/01/1619948832/epoch9999_d.pkl'
        pkl_path_g = '/home/congen/code/geoml_gan/models/two_moon/SNGAN/01/1619948832/epoch9999_g.pkl'
        # self.disc.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_epoch299_d.pkl')))
        self.disc.load_state_dict(torch.load(pkl_path_d))
        self.gen.load_state_dict(torch.load(pkl_path_g))
        # self.gen.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_epoch299_g.pkl')))

    def compute_pr(self):
        results_dir = os.path.join(
            self.result_dir + '/' + self.dataset + '/' + self.model_name + '/compute_pr/' + '%03d' % self.d)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        pkl_path_d = 'models/two_moon/SNGAN/002_5dnobnH/SNGAN_epoch4999_d.pkl'
        pkl_path_g = 'models/two_moon/SNGAN/002_5dnobnH/SNGAN_epoch4999_g.pkl'
        self.disc.load_state_dict(torch.load(pkl_path_d))
        self.gen.load_state_dict(torch.load(pkl_path_g))
        self.disc.eval()
        self.gen.eval()
        # with torch.no_grad():
        #     samples = self.gen(self.sample_z_)
        #     samples=samples.view(self.batch_size,1,28,28)
        #     grid = vutils.make_grid(samples.data[:self.batch_size], scale_each=True, range=(-1, 1), normalize=True)
        #     vutils.save_image(grid,
        #                       os.path.join(results_dir, self.model_name + '499.png'))

        # B = np.float32(ortho_group.rvs(size=1, dim=self.d, random_state=1)[:, 0:2])
        B = torch.eye(2).cuda()
        # B = torch.Tensor(B).cuda()
        z_code = self.sample_z_.mm(B)
        minz, _ = z_code.min(dim=0)  # d
        maxz, _ = z_code.max(dim=0)
        alpha = 0.1 * (maxz - minz)  # d
        minz -= alpha  # d
        maxz += alpha  # d
        ran0 = torch.linspace(minz[0].item(), maxz[0].item(), 50, device=self.device)
        ran1 = torch.linspace(minz[1].item(), maxz[1].item(), 50, device=self.device)
        Mx, My = torch.meshgrid(ran0, ran1)
        Mxy = torch.cat((Mx.t().reshape(-1, 1), My.t().reshape(-1, 1)), dim=1)  # (meshsize^2)x2
        Mxy_z = Mxy.mm(B.t())
        # mean=Mxy_z.mean([0])
        # Mxy_z = Mxy_z - mean
        # Mxy_z = Mxy_z.div(Mxy_z.norm(2, dim=[-1]).unsqueeze(-1))
        #          with torch.no_grad():
        J_list = []
        Dx_list = []
        for i in range(Mxy.shape[0] // 50):
            X, J = self.gen(Mxy_z[i * 50:(i + 1) * 50], True)
            Dx = torch.sigmoid(self.disc(X))
            J_list.append(J)
            Dx_list.append(Dx)
        J = torch.cat(J_list, 0)
        Dx = torch.cat(Dx_list, 0).squeeze()
        jtj = torch.bmm(torch.transpose(J, -2, -1), J)
        JTJ = (-0.5 * torch.slogdet(jtj)[1]).exp()
        loc = torch.zeros(Mxy_z.shape[-1])
        scale = torch.ones(Mxy_z.shape[-1])
        normal = td.Normal(loc, scale)
        diagn = td.Independent(normal, 1).log_prob(Mxy_z.squeeze().cpu()).exp()
        pg = JTJ * diagn.cuda()
        pr = (Dx * pg / (1 - Dx))
        pr = pr.reshape(Mx.shape)
        pg = pg.reshape(Mx.shape)
        JTJ = JTJ.reshape(Mx.shape)
        Dx = Dx.reshape(Mx.shape)
        z1 = torch.Tensor([-0.5, 2.5]).cuda()
        z2 = torch.Tensor([-1, 2.5]).cuda()
        t = torch.linspace(0, 1, 20, device=self.device)
        C_linear = (1 - t).unsqueeze(1) * z1 + t.unsqueeze(1) * z2
        C_linear_z = C_linear.mm(B.t())
        # C_linear_z = C_linear_z - mean
        # C_linear_z = C_linear_z.div(C_linear_z.norm(2, dim=[-1]).unsqueeze(-1))
        Gz = self.gen(C_linear_z)
        Gz = Gz.view(20, 1, 28, 28)
        grid = vutils.make_grid(Gz.data[:20], scale_each=True, range=(-1, 1), normalize=True)
        vutils.save_image(grid,
                          os.path.join(results_dir, 'Gz.png'))
        plt.imshow(Dx, extent=(ran0[0].item(), ran0[-1].item(), ran1[0].item(), ran1[-1].item()), origin='lower')
        plt.colorbar()
        plt.show()
        plt.close()
        plt.imshow(pg, extent=(ran0[0].item(), ran0[-1].item(), ran1[0].item(), ran1[-1].item()), origin='lower')
        plt.colorbar()
        plt.show()
        plt.close()
        plt.imshow(pr, extent=(ran0[0].item(), ran0[-1].item(), ran1[0].item(), ran1[-1].item()), origin='lower')
        plt.scatter(C_linear.cpu()[:, 0], C_linear.cpu()[:, 1], c='r', s=2)
        plt.colorbar()
        plt.show()
        plt.close()

    def compute_ebmpr(self):

        pkl_path_d = '/home/cong/code/geoml_gan/models/25gaussians/EBM_0gp/01/1623190533/epoch279_d.pkl'
        pkl_path_g = '/home/cong/code/geoml_gan/models/25gaussians/EBM_0gp/01/1623190533/epoch279_g.pkl'
        self.disc.load_state_dict(torch.load(pkl_path_d))
        self.gen.load_state_dict(torch.load(pkl_path_g))
        self.disc.eval()
        self.gen.eval()
        if args.dataset == 'two_moon':
            data, y = sklearn.datasets.make_moons(n_samples=self.batch_size, noise=0.1)
        elif args.dataset == 'swiss_roll':
            data = sklearn.datasets.make_swiss_roll(n_samples=self.batch_size, noise=1.0)[0]
            data = data.astype("float32")[:, [0, 2]]
            data /= 5
        elif args.dataset == '25gaussians':
            data = []
            for j in range(1000 // 25):
                for x in range(-2, 3):
                    for y in range(-2, 3):
                        point = np.random.randn(2) * 0.05
                        point[0] += 2 * x
                        point[1] += 2 * y
                        data.append(point)
            data = np.array(data[:self.batch_size], dtype='float32')
            np.random.shuffle(data)
            data /= 2.828
        points = torch.Tensor(data).cuda()
        samples = self.gen(self.sample_z_)

        samples = samples.detach().cpu().numpy()
        points = points.detach().cpu().numpy()
        plt.clf()
        ax = plt.subplot(1, 4, 1, aspect="equal", title='gen')
        ax.scatter(samples[:, 0], samples[:, 1], s=1)

        ax = plt.subplot(1, 4, 2, aspect="equal", title='data')
        ax.scatter(points[:, 0], points[:, 1], s=1)
        self.disc.cpu()

        ax = plt.subplot(1, 4, 3, aspect="equal")
        self.plt_toy_density(lambda x: -self.disc(x), ax,
                             low=-4, high=4,
                             title="p(x)")

        ax = plt.subplot(1, 4, 4, aspect="equal")
        self.plt_toy_density(lambda x: -self.disc(x), ax,
                             low=-4, high=4,
                             exp=False, title="log p(x)")

        plt.show()
        plt.close()

        # self.log_figure(fig=fig2, dir=results_dir, saved_file='Dx')

    def plot_pr(self):

        pkl_path_d = '/home/cong/code/geoml_gan/models/25gaussians/EBM_0gp/01/1621676285/epoch319_d.pkl'
        pkl_path_g = '/home/cong/code/geoml_gan/models/25gaussians/EBM_0gp/01/1621676285/epoch319_g.pkl'
        self.disc.load_state_dict(torch.load(pkl_path_d))
        self.gen.load_state_dict(torch.load(pkl_path_g))
        self.disc.eval()
        self.gen.eval()
        sample_z_ = torch.randn((1000, self.d)).to(self.device)
        if args.dataset == 'two_moon':
            data, y = sklearn.datasets.make_moons(n_samples=1000, noise=0.1)
        elif args.dataset == 'swiss_roll':
            data = sklearn.datasets.make_swiss_roll(n_samples=1000, noise=1.0)[0]
            data = data.astype("float32")[:, [0, 2]]
            data /= 5
        elif args.dataset == '25gaussians':
            data = []
            for j in range(1000 // 25):
                for x in range(-2, 3):
                    for y in range(-2, 3):
                        point = np.random.randn(2) * 0.05
                        point[0] += 2 * x
                        point[1] += 2 * y
                        data.append(point)
            data = np.array(data[:self.batch_size], dtype='float32')
            np.random.shuffle(data)
            data /= 2.828
        points = torch.Tensor(data).cuda()
        samples = self.gen(sample_z_)

        samples = samples.detach().cpu().numpy()
        points = points.detach().cpu().numpy()
        plt.clf()
        # ax = plt.subplot(1, 4, 1, aspect="equal", title='gen')
        fig = plt.figure(figsize=(2, 2))
        plt.scatter(samples[:, 0], samples[:, 1], s=1)
        plt.xticks([])
        plt.yticks([])
        fig.savefig("%s/gene_%s.png" % (self.result_dir, args.dataset), bbox_inches='tight', pad_inches=0.03)
        plt.close()
        # ax.imshow(vmax=1.5,vmin=-1.5)
        # ax = plt.subplot(1, 4, 2, aspect="equal", title='data')
        fig = plt.figure(figsize=(2, 2))
        plt.scatter(points[:, 0], points[:, 1], s=1)
        plt.xticks([])
        plt.yticks([])
        fig.savefig("%s/real_%s.png" % (self.result_dir, args.dataset), bbox_inches='tight', pad_inches=0.03)
        plt.close()
        # #ax.imshow(vmax=1.5, vmin=-1.5)
        # self.disc.cpu()
        fig = plt.figure(figsize=(2, 2))
        npts = 100
        side = np.linspace(-4, 4, npts)
        side2 = np.linspace(-4, 4, npts)
        # ax = plt.subplot(1, 4, 3, aspect="equal")
        x = np.asarray(np.meshgrid(side, side2)).transpose(1, 2, 0).reshape((-1, 2))
        x = torch.from_numpy(x).type(torch.float32).to(device)
        logpx = -self.disc(x).squeeze()

        logpx = logpx * 0.1
        logpx = logpx - logpx.logsumexp(0)
        # logpx = logpx - logpx.mean(0)
        px = np.exp(logpx.cpu().detach().numpy()).reshape(npts, npts)
        px = px / px.sum()
        plt.imshow(px, origin='lower', aspect="auto")
        plt.xticks([])
        plt.yticks([])
        fig.savefig("%s/pr_%s.png" % (self.result_dir, args.dataset), bbox_inches='tight', pad_inches=0.03)
        plt.close()

        # self.log_figure(fig=fig2, dir=results_dir, saved_file='Dx')

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
        self.model_name = args.gan_type
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
        jtj = torch.bmm(torch.transpose(J, -2, -1), J)
        detjtj = 0.5 * torch.slogdet(jtj)[1]
        logpGz = logpz - detjtj
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

        if fix:
            """ fixed noise """
            samples = self.gen(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.randn((points.shape[0], self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.gen(sample_z_)
        samples = samples.detach().cpu().numpy()
        points = points.detach().cpu().numpy()
        plt.clf()
        ax = plt.subplot(1, 4, 1, aspect="equal", title='gen')
        ax.scatter(samples[:, 0], samples[:, 1], s=1)

        ax = plt.subplot(1, 4, 2, aspect="equal", title='data')
        ax.scatter(points[:, 0], points[:, 1], s=1)
        self.disc.cpu()

        ax = plt.subplot(1, 4, 3, aspect="equal")
        self.plt_toy_density(lambda x: -self.disc(x), ax,
                             low=-4, high=4,
                             title="p(x)")

        ax = plt.subplot(1, 4, 4, aspect="equal")
        self.plt_toy_density(lambda x: -self.disc(x), ax,
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

        pkl_path_d = '/home/cong/code/geoml_gan/models/two_moon/EBM_upper_gp/01/1627600750/epoch139999_d.pkl'
        pkl_path_g = '/home/cong/code/geoml_gan/models/two_moon/EBM_upper_gp/01/1627600750/epoch139999_g.pkl'
        self.disc.load_state_dict(torch.load(pkl_path_d))
        self.gen.load_state_dict(torch.load(pkl_path_g))
        self.disc.eval()
        self.gen.eval()
        if args.dataset == 'two_moon':
            data, y = sklearn.datasets.make_moons(n_samples=1000, noise=0.1)
        elif args.dataset == 'swiss_roll':
            data = sklearn.datasets.make_swiss_roll(n_samples=1000, noise=1.0)[0]
            data = data.astype("float32")[:, [0, 2]]
            data /= 5
        elif args.dataset == '25gaussians':
            data = []
            for j in range(1000 // 25):
                for x in range(-2, 3):
                    for y in range(-2, 3):
                        point = np.random.randn(2) * 0.05
                        point[0] += 2 * x
                        point[1] += 2 * y
                        data.append(point)
            data = np.array(data[:self.batch_size], dtype='float32')
            np.random.shuffle(data)
            data /= 2.828
        points = torch.Tensor(data).cuda()
        sample_z = torch.randn((1000, self.d)).to(self.device)
        samples = self.gen(sample_z)

        samples = samples.detach().cpu().numpy()
        # fig = plt.figure(figsize=(5, 5))
        points = points.detach().cpu().numpy()
        plt.clf()
        ax = plt.subplot(1, 4, 1, aspect="equal", title='gen')
        ax.scatter(samples[:, 0], samples[:, 1], s=1)
        # fig.savefig("%s/gene.png" % (self.result_dir), bbox_inches='tight', pad_inches=0.01)
        # ax.imshow(vmax=1.5,vmin=-1.5)
        ax = plt.subplot(1, 4, 2, aspect="equal", title='data')
        plt.scatter(points[:, 0], points[:, 1], s=1)
        # fig.savefig("%s/real.png" % (self.result_dir), bbox_inches='tight', pad_inches=0.01)
        # #ax.imshow(vmax=1.5, vmin=-1.5)
        self.disc.cpu()
        #
        ax = plt.subplot(1, 4, 3, aspect="equal")
        self.plt_toy_density(lambda x: -self.disc(x), ax,
                             low=-4, high=4,
                             title="p(x)")

        ax = plt.subplot(1, 4, 4, aspect="equal")
        self.plt_toy_density(lambda x: -self.disc(x), ax,
                             low=-4, high=4,
                             exp=False, title="log p(x)")

        plt.show()
        # fig.savefig("%s/gene.png" % (self.result_dir), bbox_inches='tight', pad_inches=0.01)
        plt.close()

        # self.log_figure(fig=fig2, dir=results_dir, saved_file='Dx')

    def plot_pr(self):

        pkl_path_d = '/home/cong/code/geoml_gan/models/swiss_roll/EBM_upper_gp/01/1620981536/epoch199999_d.pkl'
        pkl_path_g = '/home/cong/code/geoml_gan/models/swiss_roll/EBM_upper_gp/01/1620981536/epoch199999_g.pkl'
        self.disc.load_state_dict(torch.load(pkl_path_d))
        self.gen.load_state_dict(torch.load(pkl_path_g))
        self.disc.eval()
        self.gen.eval()
        sample_z_ = torch.randn((1000, self.d)).to(self.device)
        if args.dataset == 'two_moon':
            data, y = sklearn.datasets.make_moons(n_samples=1000, noise=0.1)
        elif args.dataset == 'swiss_roll':
            data = sklearn.datasets.make_swiss_roll(n_samples=1000, noise=1.0)[0]
            data = data.astype("float32")[:, [0, 2]]
            data /= 5
        elif args.dataset == '25gaussians':
            data = []
            for j in range(1000 // 25):
                for x in range(-2, 3):
                    for y in range(-2, 3):
                        point = np.random.randn(2) * 0.05
                        point[0] += 2 * x
                        point[1] += 2 * y
                        data.append(point)
            data = np.array(data[:self.batch_size], dtype='float32')
            np.random.shuffle(data)
            data /= 2.828
        points = torch.Tensor(data).cuda()
        samples = self.gen(sample_z_)

        samples = samples.detach().cpu().numpy()
        points = points.detach().cpu().numpy()
        plt.clf()
        # ax = plt.subplot(1, 4, 1, aspect="equal", title='gen')
        fig = plt.figure(figsize=(2, 2))
        plt.scatter(samples[:, 0], samples[:, 1], s=1)
        plt.xlim((-4, 4))
        plt.ylim((-4, 4))
        plt.xticks([])
        plt.yticks([])
        fig.savefig("%s/gene_%s.png" % (self.result_dir, args.dataset), bbox_inches='tight', pad_inches=0.03)
        plt.close()

        # ax.imshow(vmax=1.5,vmin=-1.5)
        # ax = plt.subplot(1, 4, 2, aspect="equal", title='data')
        fig = plt.figure(figsize=(2, 2))
        plt.scatter(points[:, 0], points[:, 1], s=1)
        plt.xlim((-4, 4))
        plt.ylim((-4, 4))
        plt.xticks([])
        plt.yticks([])
        fig.savefig("%s/real_%s.png" % (self.result_dir, args.dataset), bbox_inches='tight', pad_inches=0.03)
        plt.close()
        # #ax.imshow(vmax=1.5, vmin=-1.5)
        # self.disc.cpu()
        fig = plt.figure(figsize=(2, 2))
        npts = 100
        side = np.linspace(-4, 4, npts)
        side2 = np.linspace(-4, 4, npts)
        # ax = plt.subplot(1, 4, 3, aspect="equal")
        x = np.asarray(np.meshgrid(side, side2)).transpose(1, 2, 0).reshape((-1, 2))
        x = torch.from_numpy(x).type(torch.float32).to(device)
        logpx = -self.disc(x).squeeze()

        logpx = logpx * 0.1
        logpx = logpx - logpx.logsumexp(0)
        # logpx = logpx - logpx.mean(0)
        px = np.exp(logpx.cpu().detach().numpy()).reshape(npts, npts)
        px = px / px.sum()
        plt.imshow(px, origin='lower', aspect="auto")
        plt.xticks([])
        plt.yticks([])
        fig.savefig("%s/pr_%s.png" % (self.result_dir, args.dataset), bbox_inches='tight', pad_inches=0.03)
        plt.close()

        # self.log_figure(fig=fig2, dir=results_dir, saved_file='Dx')

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
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    os.environ['CUDA_HOME'] = '/opt/cuda/cuda-10.2'
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--gan_type', type=str, default='EBM_BB',
                        choices=['EBM','EBM_OGP'],
                        help='The type of GAN')
    parser.add_argument('--dataset', type=str, default='25gaussians',
                        choices=['swiss_roll',  'two_moon', '25gaussians'],
                        help='The name of dataset')
    parser.add_argument('--mode', type=str, default='train', help='mode')
    parser.add_argument('--iters', type=int, default=150000, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=200, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=2, help='The size of input image')
    parser.add_argument('--energy_model_iters', type=int, default=1)
    parser.add_argument("--generator_iters", type=int, default=1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", type=str, default='Adam')
    parser.add_argument('--train_mode', type=str, default='mins', help='mode')
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
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--benchmark_mode', type=bool, default=True)

    args = parser.parse_args()
    # --save_dir
    if args.mode == 'train' and utils.is_debugging() == False:
        time = int(time.time())
        args.save_dir = os.path.join(args.save_dir + '/' + args.dataset + '/'
                                     + args.gan_type + '/%02d' % args.energy_model_iters + '/%03d' % time)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # --result_dir
        args.result_dir = os.path.join(args.result_dir + '/' + args.dataset + '/'
                                       + args.gan_type + '/%02d' % args.energy_model_iters + '/%03d' % time)
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)

        # --result_dir
        args.log_dir = os.path.join(args.log_dir + '/' + args.dataset + '/'
                                    + args.gan_type + '/%02d' % args.energy_model_iters + '/tb_%03d' % time)
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
    label_thresh = 2  # include only a subset of MNIST classes
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model = EBM_BB(args, layers, device)

    if args.mode == 'train':
        model.trainer(args.iters)
    # model.refineD(train_loader, args.epoch)
    elif args.mode == 'ebmpr':
        model.compute_ebmpr()
    elif args.mode == 'plot_pr':
        model.plot_pr()




