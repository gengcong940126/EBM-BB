import torch
import torch.nn as nn
from geoml import nnj
from geoml import *
import time
import json
import pickle
import math
import collections
from tqdm import tqdm
import os, argparse
import torch.nn.functional as F
from scipy.stats import ortho_group
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from score.inception_network import InceptionV3
from score.fid_score import calculate_frechet_distance
from score.inception_score import kl_scores
from score.F_beta import calculate_f_beta_score,plot_pr_curve
from score.score import compute_is_and_fid
import numpy as np
from torchvision import datasets, transforms
import torchplot as plt
from sklearn.cluster import KMeans
import torch.distributions as td
import random


class InfiniteDataLoader(object):
    """docstring for InfiniteDataLoader"""

    def __init__(self, dataloader):
        super(InfiniteDataLoader, self).__init__()
        self.dataloader = dataloader
        self.data_iter = None

    def next(self):
        try:
            data = self.data_iter.next()
        except Exception:
            # Reached end of the dataset
            self.data_iter = iter(self.dataloader)
            data = self.data_iter.next()

        return data

    def __len__(self):
        return len(self.dataloader)
class EnergyModel(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
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
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.expand = nn.utils.spectral_norm(nn.Linear(4 * 4 * dim, 1))
        #self.apply(weights_init)

    def forward(self, x):
        out = self.main(x).view(x.size(0), -1)
        return self.expand(out).squeeze(-1)
class EnergyModel2(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
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
        #self.apply(weights_init)

    def forward(self, x):
        out = self.main(x).view(x.size(0), -1)
        return self.expand(out).squeeze(-1)
class Generator(nn.Module):
    def __init__(self, z_dim=128, dim=512):
        super().__init__()
        self.expand = nn.Linear(z_dim, 4 * 4 * dim)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, 4, 2, 1),
            nn.BatchNorm2d(dim // 2,momentum=0.1),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim // 2, dim // 4, 4, 2, 1),
            nn.BatchNorm2d(dim // 4,momentum=0.1),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim // 4, dim // 8, 4, 2, 1),
            nn.BatchNorm2d(dim // 8,momentum=0.1),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim // 8, 3, 3, 1, 1),
            nn.Tanh(),
        )
        #self.apply(weights_init)

    def forward(self, z):
        out = self.expand(z).view(z.size(0), -1, 4, 4)
        return self.main(out)

class EBM(nn.Module):
    def __init__(self, args, device='cpu'):
        super(EBM, self).__init__()
        self.device = device
        self.d = args.z_dim
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
        if args.sn==True:
            self.disc = EnergyModel()
        else:
            self.disc = EnergyModel2()
        self.gen = Generator(self.d)

        if self.gpu_mode:
            self.gen.cuda()
            self.disc.cuda()
        self.to(self.device)

        if args.ngpu > 1:
            gpu_ids = range(args.ngpu)
            self.gen=nn.DataParallel(self.gen,device_ids=gpu_ids)
            self.disc = nn.DataParallel(self.disc, device_ids=gpu_ids)
    def trainer(self, data_loader, iters=50):
        #self.load()
        optimizer_d = torch.optim.Adam(self.disc.parameters(), betas=(0.0, 0.9), lr=args.lrd)
        optimizer_g = torch.optim.Adam(self.gen.parameters(), betas=(0.0, 0.9), lr=args.lrg)
        d_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=args.milestone, gamma=args.gammad)
        g_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_g, milestones=args.milestone, gamma=args.gammag)
        sum_loss_d = 0
        sum_loss_g = 0
        writer = SummaryWriter(log_dir=self.log_dir)
        for iteration in range(iters):
            epoch = (iteration) // len(data_loader)
            e_costs = []
            g_costs = []
            # for batch_idx, (data,) in enumerate(data_loader):
            for i in range(args.energy_model_iters):
                optimizer_d.zero_grad()
                data = data_loader.next()[0].cuda()
                data = data.to(self.device)
                data.requires_grad_()
                # discriminator
                # for i in range(n_disc):
                z_train = torch.randn((data.shape[0], self.d))
                if self.gpu_mode:
                    z_train = z_train.cuda()
                    #torch.save(z_train, os.path.join(self.save_dir, 'z_train.pkl'))
                    #torch.save(self.disc.state_dict(), os.path.join(self.save_dir, 'd.pkl'))
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
                # L2 norm
                gp_loss = (gradients.norm(2, dim=1) ** 2).mean()
                D_loss = (d_real - d_fake).mean() + gp_loss * args.gp_weight
                # D_loss2 = (-d_real2 + d_fake2).mean()
                e_costs.append([d_real.mean().item(), d_fake.mean().item(),D_loss.item(), gp_loss.item()])
                D_loss.backward()
                optimizer_d.step()
            d_real_mean, d_fake_mean,  D_loss_mean, gp_loss_mean= np.mean(e_costs[-args.energy_model_iters:], 0)
            sum_loss_d += D_loss_mean.item() * len(data)

            # generator
            for i in range(args.generator_iters):
                optimizer_g.zero_grad()
                z_train = torch.randn((data.shape[0], self.d))
                if self.gpu_mode:
                    z_train = z_train.cuda()
                    z_train.requires_grad_()
                fake= self.gen(z_train)
                d_fake_g = self.disc(fake)
                #J=self.fd_jacobian(self.gen,z_train)
                #J = self.compute_jacobian(fake,z_train)
                #J = self.batch_jacobian(z_train)
                # loc = torch.zeros(z_train.shape[-1])
                # scale = torch.ones(z_train.shape[-1])
                # normal = td.Normal(loc, scale)
                # logpz = td.Independent(normal, 1).log_prob(z_train.squeeze().cpu())
                #jtj = torch.bmm(J,torch.transpose(J, -2, -1))
                #JTJ = (-0.5 * torch.slogdet(jtj)[1])
                # H=-(logpz.cuda()+JTJ)
                #H = - JTJ
                H,mu=self.compute_entropy_s(z_train,iteration)
                #H, mu = self.compute_entropy_adam(z_train)
                g_loss = d_fake_g.mean()-H.mean()*args.H_weight
                g_costs.append([d_fake_g.mean().item(), g_loss.item(), H.mean().item(),\
                                mu.mean().item()])
                g_loss.backward()
                optimizer_g.step()
            d_fake_g_mean, g_loss_mean, H_mean, mu_mean\
                = np.mean(g_costs[-args.generator_iters:], 0)
            sum_loss_g += g_loss_mean.item() * len(data)
            if (iteration+1) % len(data_loader) ==0:
                d_lr_scheduler.step()
                g_lr_scheduler.step()
            if iteration % 250 == 0:
                writer.add_scalars('d_logit_mean', {'r_logit_mean': d_real_mean,
                                                    'f_logit_mean': d_fake_mean,
                                                    'G_f_logit_mean': d_fake_g_mean}, iteration)
                writer.add_scalars('singular_value', {'s_mean': mu_mean}, iteration)
                writer.add_scalar('D_loss', D_loss_mean, iteration)
                writer.add_scalar('g_loss', g_loss_mean, iteration)
                writer.add_scalar('gp_loss', gp_loss_mean, iteration)
                writer.add_scalar('H', H_mean, iteration)
            if iteration % (len(data_loader)) == 0:
                with torch.no_grad():
                    self.visualize_results((epoch))
            if (iteration) % (len(data_loader)) == 0:
                avg_loss_d = sum_loss_d / len(data_loader) / args.batch_size
                avg_loss_g = sum_loss_g / len(data_loader) / args.batch_size
                print('(MEAN) ====> Epoch: {} Average loss d: {:.4f}'.format(epoch, avg_loss_d))
                print('(MEAN) ====> Epoch: {} Average loss g: {:.4f}'.format(epoch, avg_loss_g))
                sum_loss_d = 0
                sum_loss_g = 0
            if (iteration+1) % (20*len(data_loader)) == 0:
                self.save(epoch)
            # if batch_idx % 50 == 0:
            # summary_defaultdict2txtfig(default_dict=summary_d, prefix='GAN', step=epoch*len(data_loader) + batch_idx,
            # textlogger=self.myargs.textlogger, save_fig_sec=60)

    def fd_jacobian(self,function, x, h=1e-4):
        """Compute finite difference Jacobian of given function
        at a single location x. This function is mainly considered
        useful for debugging."""

        no_batch = x.dim() == 1
        if no_batch:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            raise Exception("The input should be a D-vector or a BxD matrix")
        B, D = x.shape

        # Compute finite differences
        E = h * torch.eye(D)
        E=E.to(self.device)
        try:
            # Disable "training" in the function (relevant eg. for batch normalization)
            orig_state = function.eval()
            Jnum = torch.cat([((function(x[b] + E) - function(x[b].unsqueeze(0))) / h).unsqueeze(0)
                              for b in range(B)])
            Jnum=Jnum.view(Jnum.shape[0],Jnum.shape[1],-1)
        finally:
            function.train()  # re-enable training

        if no_batch:
            Jnum = Jnum.squeeze(0)

        return Jnum
    def compute_entropy(self,fake):
        fake=fake.view(fake.shape[0],-1)
        pz=2/(1+fake.norm(2,-1)).unsqueeze(-1)
        s=torch.cat((fake*pz,1-pz),-1)
        logps=s.mm((s.mean(0)/(s.mean(0).norm(2))).unsqueeze(-1))
        logjtj=torch.log(pz)
        logpg=logps+logjtj
        H=-logpg.mean()
        return H
    def compute_entropy_s(self, z,iter,ds=1):
        self.gen.eval()
        v = torch.randn(z.shape).to(self.device)
        p = torch.randn(z.shape).to(self.device)
        fake_eval=self.gen(z)
        projection = torch.ones_like(fake_eval, requires_grad=True)
        intermediate = torch.autograd.grad(fake_eval, z, projection, create_graph=True)
        Jv = torch.autograd.grad(intermediate[0], projection, v, create_graph=True)[0]
        #Jv=J.bmm(self.v.unsqueeze(-1)).squeeze()
        mu=Jv.norm(2,dim=(1,2,3))/v.norm(2,dim=-1)
        for i in range(2):
            JTJv=(Jv.detach()*fake_eval).sum((1,2,3),True)
            r=torch.autograd.grad(JTJv, z, torch.ones_like(JTJv, requires_grad=True), retain_graph=True)[0]\
               -(mu**2).unsqueeze(-1)*v
            #r = J.permute(0,2,1).bmm(Jv.unsqueeze(-1)).squeeze(-1)-mu.unsqueeze(-1)*self.v
            #self.v.data.copy_(self.RaRitz(self.v,r,intermediate,projection).squeeze(-1))
            v,p=self.RaRitz(v, r,p,intermediate, projection)
            Jv = torch.autograd.grad(intermediate[0], projection, v.detach(), create_graph=True)[0]
            #Jv = J.bmm(self.v.unsqueeze(-1)).squeeze()
            mu = Jv.norm(2, dim=(1,2,3))
        # if self.v.norm(2, dim=-1).min()==0:
        #     print(iter)
        #     print(self.v)
        where_is_nan = torch.isnan(mu)
        if len(mu[where_is_nan]) > 0:
            print(iter)
            print(Jv.norm(2, dim=(1,2,3)))
            print(self.v.norm(2, dim=-1))
            print(mu)
            return
        # else:
        #     torch.save(self.gen.state_dict(), os.path.join(self.save_dir, 'g.pkl'))
        #     torch.save(self.v, os.path.join(self.save_dir, 'v.pkl'))
        #     torch.save(z, os.path.join(self.save_dir, 'z.pkl'))
        est = (self.d / ds) * torch.log(mu)
        H = est.unsqueeze(-1)
        self.gen.train()
        return H,mu
    def compute_entropy_adam(self, z,ds=1):
        self.gen.eval()
        #fake2=self.gen(z)
        #self.gen.disable_training()
        #a=torch.randn((self.batch_size,self.d),requires_grad=True).cuda()
        fake_eval=self.gen(z)
        projection = torch.ones_like(fake_eval, requires_grad=True)
        intermediate = torch.autograd.grad(fake_eval, z, projection, create_graph=True)
        for i in range(2):
            Jv = torch.autograd.grad(intermediate[0], projection, self.v_ada, create_graph=True)[0]
            size = len(Jv.shape)
            # Jv=J.bmm(self.v.unsqueeze(-1)).squeeze()
            mu = Jv.norm(2, dim=list(range(1, size))) / self.v_ada.norm(2, dim=-1)
            loss=mu.sum()
            for name, param in self.gen.named_parameters():
                param.requires_grad = False
            loss.backward(retain_graph=True)
            self.optimizer_ada.step()
            for name, param in self.gen.named_parameters():
                param.requires_grad = True
        Jv = torch.autograd.grad(intermediate[0], projection, self.v_ada, create_graph=True)[0]
        #Jv = J.bmm(self.v.unsqueeze(-1)).squeeze()
        mu = Jv.norm(2, dim=list(range(1,size))) / (self.v_ada.norm(2, dim=-1))
        est = (self.d / ds) * torch.log(mu)
        H = est.unsqueeze(-1)
        #s_gt=torch.svd(J).S[:, -1]
        self.gen.train()
        return H,mu

    def RaRitz(self,v,r,p,intermediate, projection):
        JV=[]
        r=r/r.mean(-1,True)
        # v=v/v.mean(-1,True)
        # r=r-((v*r).sum(-1,True))/((v*v).sum(-1,True))*v
        # v=v/v.norm(2,1,True)
        # r=r/(r.norm(2,1,True))
        if p.norm(2,1).min()==0:
            p=torch.randn((self.batch_size, self.d))
        V=torch.stack((v,r,p),-1)
        #V=V.bmm(torch.svd(V).V)
        #V=V/V.norm(2,1,True)
        V =torch.svd(V).U
        #V,_ = torch.qr(V)
        #V=torch.stack((v,r),0)
        for i in range(V.shape[-1]):
            Jv = torch.autograd.grad(intermediate, projection, V[:,:,i], retain_graph=True)[0]
            JV.append(Jv)
        #JV=J.bmm(V)
        if len(JV[0].shape)==4:
            JV=torch.stack(JV,-1).flatten(1, 3)
        else:
            JV = torch.stack(JV, -1)
        #JV_gt=J.bmm(V)
        #vjjv=JV.permute(0,2,1).bmm(JV)
        v_min=torch.svd(JV).V[:, :,-1:]
        p_op=V[:, :, -2:].bmm(v_min[:, -2:]).squeeze(-1)
        p_op_norm=p_op.norm(2, dim=-1)
        p.data.index_copy_(0, torch.where(~p_op_norm.isnan())[0], p_op[~p_op_norm.isnan()].detach())
        v_op=V.bmm(v_min).squeeze(-1)
        v_op_norm = v_op.norm(2, dim=-1)
        v.data.index_copy_(0, torch.where(~v_op_norm.isnan())[0], v_op[~v_op_norm.isnan()].detach())
        return v,p
    def visualize_results(self, epoch, fix=True):
        self.gen.eval()
        self.disc.eval()

        if fix:
            """ fixed noise """
            samples = self.gen(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.randn((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.gen(sample_z_)

        if self.gpu_mode:
            samples = samples.view(self.batch_size, 3, 32, 32)
            # samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.view(self.batch_size, 3, 32, 32)
            # samples = samples.data.numpy().transpose(0, 2, 3, 1)

        # samples = (samples + 1) / 2
        grid = vutils.make_grid(samples.data[:self.batch_size], scale_each=True, range=(-1, 1), normalize=True)
        vutils.save_image(grid, self.result_dir +'/' + '_d_epoch%03d' % epoch + '.png')
        self.gen.train()
        self.disc.train()

        # grid2 = torchvision.utils.make_grid(samples.data, scale_each=True, range=(0, 1), normalize=True)
        # train_writer.add_image('generate', grid, epoch * 1875 + batch_idx)
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
        # with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
        #     pickle.dump(self.train_hist, f)

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

    def compute_jacobian(self,outputs, inputs, create_graph=True):

        jac = outputs.new_zeros(outputs.size() + inputs.size()[1:]).view((outputs.shape[0],-1,) + inputs.size()[1:])
        for i in range(jac.shape[1]):
            # print(out.requires_grad)
            # print(inputs.requires_grad)
            # input("IN various.calculate_jacobian()")

            col_i = torch.autograd.grad(outputs.view(outputs.shape[0],-1)[:,i:i+1], inputs,
                                        grad_outputs=torch.ones([outputs.shape[0],1], device=device),
                                        retain_graph=True,create_graph=create_graph, allow_unused=True)[0]

            jac[:,i,:] = col_i
        if create_graph:
            jac.requires_grad_()
        return jac
    #

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
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--gan_type', type=str, default='EBM',
                        choices=['GAN', 'CGAN', 'infoGAN', 'ACGAN', 'EBGAN', 'BEGAN', 'WGAN', 'WGAN_GP', 'DRAGAN',
                                 'LSGAN'], help='The type of GAN')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'svhn', 'stl10', 'lsun-bed'],
                        help='The name of dataset')
    parser.add_argument('--mode', type=str, default='ebmpr', help='mode')
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--bn", type=bool, default=True)
    parser.add_argument("--sn", type=bool, default=False)
    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=128, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=32, help='The size of input image')
    parser.add_argument("--lrd", type=int, default=2e-4)
    parser.add_argument("--lrg", type=int, default=2e-4)
    parser.add_argument("--gammad", type=int, default=1)
    parser.add_argument("--gammag", type=int, default=1)
    parser.add_argument("--milestone", type=float, default=[70])
    parser.add_argument('--energy_model_iters', type=int, default=1)
    parser.add_argument("--generator_iters", type=int, default=1)
    parser.add_argument("--cl", type=int, default=4)
    parser.add_argument("--gp_weight", type=float, default=0.1)
    parser.add_argument("--H_weight", type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--benchmark_mode', type=bool, default=True)

    args = parser.parse_args()
    # --save_dir

    if args.mode == 'train' and is_debugging() == False:
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
    elif is_debugging() == True:
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
    setup_seed(args.seed)
    # layers = torch.linspace(28**2, 64, 3).int()
    #layers=[3072,4096,1024,512,64]
    #layers = [3, 64,128,256,512,64]
    label_thresh = 4  # include only a subset of MNIST classes
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    ## Data
    transform = transforms.Compose([transforms.Resize((args.input_size, args.input_size)), transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    test_transform = transforms.Compose(
        [transforms.Resize((args.input_size, args.input_size)),
         transforms.ToTensor(),
         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    #cifar10_train = datasets.CIFAR10('/home/cong/code/geoml_gan/data/cifar10', train=True, download=True, transform=transform)
    cifar10_train = datasets.CIFAR10('/home/congen/code/geoml_gan/data/cifar10', train=True, download=True,
                                     transform=transform)
    #x_train = transform((mnist_train.data) / 255).reshape(-1, 784)
    #y_train = mnist_train.targets
    #idx = y_train < label_thresh  # only use digits 0, 1, 2, ...
    #num_classes = y_train[idx].unique().numel()
    #x_train = x_train[idx]
    #y_train = y_train[idx]
    # x_train = x_train[idx][:subset_size]
    # y_train = y_train[idx][:subset_size]
    #N = x_train.shape[0]
    #train_data = torch.utils.data.TensorDataset(x_train)
    train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_loader = dict(train=InfiniteDataLoader(train_loader))
    # Fit model
    model = EBM(args, device)
    iters = args.epoch * len(train_loader['train'])
    # max_iter=args.epoch*len(train_loader)*args.energy_model_iters
    # pbar = zip(train_loader, range(0, max_iter))
    # iter = iter(train_loader['train'])
    #model.trainer(train_loader['train'], iters)
    cifar10_test = datasets.CIFAR10('/home/congen/code/geoml_gan/data/cifar10', train=False, download=True,
                                     transform=test_transform)
    test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=args.batch_size, shuffle=False, drop_last=False)
    model.compute_ebmpr(test_loader)
    # model.refineD(train_loader, args.epoch)
    # model.compute_pr_high()


