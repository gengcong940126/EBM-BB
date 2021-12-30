import torch
import torch.nn as nn
import time
import json
import os, argparse
import utils
import data
import torch.nn.functional as F
from scipy.stats import ortho_group
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from stochman import nnj
# from score.inception_network import InceptionV3
# from score.fid_score import calculate_frechet_distance
# from score.inception_score import kl_scores
# from score.F_beta import calculate_f_beta_score,plot_pr_curve
# from score.score import compute_is_and_fid
import numpy as np
from torchvision import datasets, transforms
import torchplot as plt
import torch.distributions as td
import random


class EnergyModel(nn.Module):
    def __init__(self, dim=512):
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
        self.l = nn.Linear(1, 1, bias=False)
    def forward(self, x):
        out = self.main(x).view(x.size(0), -1)
        out = self.l(self.expand(out)).squeeze(-1)
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

class EBM_0GP(nn.Module):
    def __init__(self, args, device='cpu'):
        super(EBM_0GP, self).__init__()
        self.device = device
        self.d = args.z_dim
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.model_name = args.gan_type
        self.sample_z_ = torch.randn((self.batch_size, self.d)).to(self.device)
        self.disc = EnergyModel()
        self.gen = Generator(self.d)
        self.to(self.device)

        if args.ngpu > 1:
            gpu_ids = range(args.ngpu)
            self.gen=nn.DataParallel(self.gen,device_ids=gpu_ids)
            self.disc = nn.DataParallel(self.disc, device_ids=gpu_ids)
    def trainer(self, data_loader, iters=50):
        optimizer_d = torch.optim.Adam(self.disc.parameters(), betas=(0.0, 0.9), lr=args.lrd)
        optimizer_g = torch.optim.Adam(self.gen.parameters(), betas=(0.0, 0.9), lr=args.lrg)
        sum_loss_d = 0
        sum_loss_g = 0
        writer = SummaryWriter(log_dir=self.log_dir)
        for iteration in range(iters):
            epoch = (iteration) // len(data_loader)
            e_costs = []
            g_costs = []
            #energy function
            for i in range(args.energy_model_iters):
                data = data_loader.next()[0].cuda()
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
                # L2 norm
                gp_loss = (gradients.norm(2, dim=1) ** 2).mean()
                D_loss = (d_real - d_fake).mean() + gp_loss * args.gp_weight
                e_costs.append([d_real.mean().item(), d_fake.mean().item(),D_loss.item(), gp_loss.item()])
                optimizer_d.zero_grad()
                D_loss.backward()
                optimizer_d.step()
            d_real_mean, d_fake_mean,  D_loss_mean, gp_loss_mean= np.mean(e_costs[-args.energy_model_iters:], 0)
            sum_loss_d += D_loss_mean.item() * len(data)

            # generator
            for i in range(args.generator_iters):
                z_train = torch.randn((data.shape[0], self.d)).to(self.device)
                z_train.requires_grad_()
                fake= self.gen(z_train)
                d_fake_g = self.disc(fake)
                if args.train_mode == 'evals':
                    H=self.compute_entropy_s(z_train)
                elif args.train_mode == 'acc':
                    H = self.compute_entropy_acc(z_train)
                g_loss = d_fake_g.mean()-H.mean()*args.H_weight
                g_costs.append([d_fake_g.mean().item(), g_loss.item(), H.mean().item()])
                optimizer_g.zero_grad()
                g_loss.backward()
                optimizer_g.step()
            d_fake_g_mean, g_loss_mean, H_mean\
                = np.mean(g_costs[-args.generator_iters:], 0)
            sum_loss_g += g_loss_mean.item() * len(data)

            if iteration % 250 == 0:
                writer.add_scalars('d_logit_mean', {'r_logit_mean': d_real_mean,
                                                    'f_logit_mean': d_fake_mean,
                                                    'G_f_logit_mean': d_fake_g_mean}, iteration)
                writer.add_scalar('D_loss', D_loss_mean, iteration)
                writer.add_scalar('g_loss', g_loss_mean, iteration)
                writer.add_scalar('gp_loss', gp_loss_mean, iteration)
                writer.add_scalar('H', H_mean, iteration)
            if iteration % (len(data_loader)) == 0:
                with torch.no_grad():
                    self.visualize_results(epoch)
            if iteration % (10*len(data_loader)) == 0:

            if (iteration) % (len(data_loader)) == 0:
                avg_loss_d = sum_loss_d / len(data_loader) / args.batch_size
                avg_loss_g = sum_loss_g / len(data_loader) / args.batch_size
                print('(MEAN) ====> Epoch: {} Average loss d: {:.4f}'.format(epoch, avg_loss_d))
                print('(MEAN) ====> Epoch: {} Average loss g: {:.4f}'.format(epoch, avg_loss_g))
                sum_loss_d = 0
                sum_loss_g = 0
            if (iteration+1) % (20*len(data_loader)) == 0:
                self.save(epoch)

    def RaRitz(self,v, r, p, intermediate, projection, device):
        JV = []
        r = r / (r.mean(-1, True))
        if p.norm(2, 1).min() == 0:
            p = torch.randn((p.shape[0], p.shape[-1]))
        V = torch.stack((v, r, p), -1)
        try:
            V = torch.svd(V).U
        except Exception as e:
            print(e)
            V = torch.randn(V.shape).to(device)
            V = torch.svd(V).U

        for i in range(V.shape[-1]):
            Jv = torch.autograd.grad(intermediate, projection, V[:, :, i], retain_graph=True)[0]
            JV.append(Jv)

        if len(JV[0].shape) == 4:
            JV = torch.stack(JV, -1).flatten(1, 3)
        else:
            JV = torch.stack(JV, -1)
        v_min = torch.svd(JV).V[:, :, -1:].cuda()
        p_op = V[:, :, -2:].bmm(v_min[:, -2:]).squeeze(-1)
        p_op_norm = p_op.norm(2, dim=-1)
        p.data.index_copy_(0, torch.where(~p_op_norm.isnan())[0], p_op[~p_op_norm.isnan()].detach())
        v_op = V.bmm(v_min).squeeze(-1)
        v_op_norm = v_op.norm(2, dim=-1)
        v.data.index_copy_(0, torch.where(~v_op_norm.isnan())[0], v_op[~v_op_norm.isnan()].detach())
        return v, p
    def compute_entropy_s(self, z):
        z.requires_grad_()
        self.gen.eval()
        fake_eval = self.gen(z)
        projection = torch.ones_like(fake_eval, requires_grad=True).to(device)
        intermediate = torch.autograd.grad(fake_eval, z, projection, create_graph=True)
        v = torch.randn(z.shape).to(device)
        p = torch.randn(z.shape).to(device)
        Jv = torch.autograd.grad(intermediate[0], projection, v, retain_graph=True)[0]
        size = len(Jv.shape)
        # Jv=J.bmm(self.v.unsqueeze(-1)).squeeze()
        mu = Jv.norm(2, dim=list(range(1, size))) / v.norm(2, dim=-1)
        for i in range(args.ssv_iter):
            JTJv = (Jv.detach() * fake_eval).sum(list(range(1, size)), True)
            r = torch.autograd.grad(JTJv, z, torch.ones_like(JTJv, requires_grad=True), retain_graph=True)[0] \
                - (mu ** 2).unsqueeze(-1) * v
            v, p = self.RaRitz(v, r, p, intermediate[0], projection, device)
            Jv = torch.autograd.grad(intermediate[0], projection, v.detach(), create_graph=True)[0]
            mu = Jv.norm(2, dim=list(range(1, size)))
        est = z.shape[-1]  * torch.log(mu)
        H = est.mean()
        return H
    def compute_entropy_acc(self, z):
        self.gen.eval()
        _, J = self.gen(z, True)
        jtj = torch.bmm(torch.transpose(J, -2, -1), J)
        H = 0.5 * torch.slogdet(jtj)[1].unsqueeze(-1)
        self.gen.train()
        return H.mean()
    def visualize_results(self, epoch, fix=True):
        self.gen.eval()
        self.disc.eval()

        if fix:
            """ fixed noise """
            samples = self.gen(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.randn((self.batch_size, self.z_dim)).to(self.device)

            samples = self.gen(sample_z_)


        samples = samples.view(self.batch_size, 3, 32, 32)
            # samples = samples.data.numpy().transpose(0, 2, 3, 1)

        # samples = (samples + 1) / 2
        grid = vutils.make_grid(samples.data[:self.batch_size], scale_each=True, range=(-1, 1), normalize=True)
        vutils.save_image(grid, self.result_dir +'/' + '_d_epoch%03d' % epoch + '.png')
        self.gen.train()
        self.disc.train()
    def calculate_metric_score(self,dataloader):
        self.disc.eval()
        self.gen.eval()
        fid_cache = '/home/congen/code/AGE/data/tf_fid_stats_cifar10_32.npz'
        n_generate = 10000
        num_split, num_run4PR, num_cluster4PR, beta4PR = 1, 10, 20, 8
        is_scores, fid_score = compute_is_and_fid(self.gen, self.d, args, device,
                                                  fid_cache, n_generate=n_generate, splits=num_split)
        precision, recall, f_beta, f_beta_inv = calculate_f_beta_score(dataloader, self.gen, self.d,
                                                                       n_generate, num_run4PR, num_cluster4PR, beta4PR,
                                                                       device)
        PR_Curve = plot_pr_curve(precision, recall, self.result_dir)

        print('IS', is_scores)
        print('FID', fid_score)
        print('F8', f_beta)
        print('F1/8', f_beta_inv)
    def compute_ebmpr(self,dataloader):
        pkl_path_d = '/home/congen/code/geoml_gan/models/cifar10/EBM/128/01/1624620823/epoch099_d.pkl'
        pkl_path_g = '/home/congen/code/geoml_gan/models/cifar10/EBM/128/01/1624620823/epoch099_g.pkl'
        self.disc.load_state_dict(torch.load(pkl_path_d))
        self.gen.load_state_dict(torch.load(pkl_path_g))
        self.disc.eval()
        self.gen.eval()
        fid_cache='/home/congen/code/AGE/data/tf_fid_stats_cifar10_32.npz'
        n_generate = 10000
        num_split, num_run4PR, num_cluster4PR, beta4PR = 1, 10, 20, 8
        is_scores, fid_score=compute_is_and_fid(self.gen, self.d, args, device,
                                                fid_cache, n_generate=n_generate,splits=num_split)
        precision, recall, f_beta, f_beta_inv = calculate_f_beta_score(dataloader, self.gen, self.d,
                                n_generate,num_run4PR, num_cluster4PR, beta4PR,device)
        PR_Curve = plot_pr_curve(precision, recall,self.result_dir)

        print('IS',is_scores)
        print('FID',fid_score)
        print('F8',f_beta)
        print('F1/8',f_beta_inv)
    def save(self, epoch):
        torch.save(self.disc.state_dict(), os.path.join(self.save_dir, 'epoch%03d' % epoch+'_d.pkl'))
        torch.save(self.gen.state_dict(), os.path.join(self.save_dir, 'epoch%03d' % epoch+'_g.pkl'))

    def load(self):
        # save_dir = os.path.join(self.save_dir, self.dataset, self.model_name,'%03d'%self.d)
        pkl_path_d = '/home/congen/code/geoml_gan/models/cifar10/EBM/128/1617651553/epoch029_d.pkl'
        pkl_path_g = '/home/congen/code/geoml_gan/models/cifar10/EBM/128/1617651553/epoch029_g.pkl'
        pkl_path_v = '/home/congen/code/geoml_gan/models/cifar10/EBM/128/1617651553/epoch029_v.pkl'
        # self.disc.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_epoch299_d.pkl')))
        self.disc.load_state_dict(torch.load(pkl_path_d))
        self.gen.load_state_dict(torch.load(pkl_path_g))
        self.v.data.copy_(torch.load(pkl_path_v))
        # self.gen.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_epoch299_g.pkl')))

    def compute_pr_high(self):
        results_dir = os.path.join(
            self.result_dir + '/' + self.dataset + '/' + self.model_name + '/compute_pr/' + '%03d' % self.d)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        pkl_path_d = 'models/mnist/GAN/refineD/064/epoch299_d.pkl'
        pkl_path_g = 'models/mnist/GAN/refineD/064/epoch299_g.pkl'
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

        B = np.float32(ortho_group.rvs(size=1, dim=self.d, random_state=1)[:, 0:2])
        # B=torch.eye(2).cuda()
        B = torch.Tensor(B).cuda()
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
        mean = Mxy_z.mean([0])
        Mxy_z = Mxy_z - mean
        Mxy_z = Mxy_z.div(Mxy_z.norm(2, dim=[-1]).unsqueeze(-1))
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
        z1 = torch.Tensor([0, 0]).cuda()
        z2 = torch.Tensor([-1, -2]).cuda()
        t = torch.linspace(0, 1, 20, device=self.device)
        C_linear = (1 - t).unsqueeze(1) * z1 + t.unsqueeze(1) * z2
        C_linear_z = C_linear.mm(B.t())
        C_linear_z = C_linear_z - mean
        C_linear_z = C_linear_z.div(C_linear_z.norm(2, dim=[-1]).unsqueeze(-1))
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


if __name__ == "__main__":
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=2
        export PORT=6007
        export CUDA_HOME=/opt/cuda/cuda-10.2
        export TIME_STR=1
        export PYTHONPATH=./:./examples
        python 	./examples/cifar_gan.py


    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    desc = "Pytorch implementation of EBM collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--gan_type', type=str, default='EBM_0GP',
                        choices=['EBM_BB', 'EBM_0GP'], help='The type of EBM')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'svhn'],
                        help='The name of dataset')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train',  'ebmpr'],help='mode')
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--bn", type=bool, default=True)
    parser.add_argument("--sn", type=bool, default=True)
    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=128, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=32, help='The size of input image')
    parser.add_argument("--lrd", type=int, default=2e-4)
    parser.add_argument("--lrg", type=int, default=2e-4)
    parser.add_argument('--energy_model_iters', type=int, default=1)
    parser.add_argument("--generator_iters", type=int, default=1)
    parser.add_argument("--ssv_iter", type=int, default=2)
    parser.add_argument("--gp_weight", type=float, default=0.001)
    parser.add_argument('--train_mode', type=str, default='acc',
                        choices=['evals', 'acc'], help='mode')
    parser.add_argument("--H_weight", type=int, default=1)
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
                + args.gan_type + '/%03d' % args.z_dim +'/%02d' % args.energy_model_iters+ '/%03d' % time)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # --result_dir
        args.result_dir = os.path.join(args.result_dir + '/' + args.dataset + '/'
            + args.gan_type + '/%03d' % args.z_dim +'/%02d' % args.energy_model_iters+ '/%03d' % time)
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)

        # --result_dir
        args.log_dir = os.path.join(args.log_dir + '/' + args.dataset + '/'
                + args.gan_type + '/%03d' % args.z_dim + '/%02d' % args.energy_model_iters+'/tb_%03d' % time)
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
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    ## Data
    transform = transforms.Compose([transforms.Resize((args.input_size, args.input_size)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    test_transform = transforms.Compose(
        [transforms.Resize((args.input_size, args.input_size)),
         transforms.ToTensor(),
         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    if args.dataset=='cifar10':
        cifar10_train = datasets.CIFAR10('/home/congen/code/geoml_gan/data/cifar10', train=True, download=True,
                                     transform=transform)

    train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_loader = dict(train=data.InfiniteDataLoader(train_loader))
    # Fit model
    model = EBM_0GP(args, device)
    if args.mode == 'train':
        iters = args.epoch * len(train_loader['train'])
        model.trainer(train_loader['train'], iters)
    elif args.mode == 'ebmpr':
        cifar10_test = datasets.CIFAR10('/home/congen/code/geoml_gan/data/cifar10', train=False, download=True,
                                     transform=test_transform)
        test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=args.batch_size, shuffle=False, drop_last=False)
        model.compute_ebmpr(test_loader)




