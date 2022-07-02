import torch
import torch.nn as nn
import time
import json
import os, argparse
import utils
import data
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from stochman import nnj
import torch.nn.functional as F
from score.F_beta import calculate_f_beta_score,plot_pr_curve
from score.score import compute_is_and_fid
import numpy as np
from torchvision import datasets, transforms
import torchplot as plt
import torch.distributions as td



@utils.register_model(name='EBM_0GP')
class EBM_0GP(nn.Module):
    def __init__(self, args, device='cpu'):
        super(EBM_0GP, self).__init__()
        self.device = device
        self.d = args.z_dim
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.model_name = args.EBM_type
        self.sample_z_ = torch.randn((self.batch_size, self.d)).to(self.device)
        self.optimizers = []
        self.epoch = 0
        if args.input_size==32:
            module = __import__('network.DCGAN', fromlist=['something'])
            self.disc = module.EnergyModel(args)
            self.gen = module.Generator(self.d)
        elif args.input_size==64:
            module = __import__('network.Resnet', fromlist=['something'])
            self.disc = module.EnergyModel(args)
            self.gen = module.Generator(args,self.d)
        with open("{}/args.txt".format(args.result_dir), 'a') as f:
            print('\n', self.disc, '\n', self.gen, file=f)
        self.to(self.device)

        if args.ngpu > 1:
            gpu_ids = range(args.ngpu)
            self.gen=nn.DataParallel(self.gen,device_ids=gpu_ids)
            self.disc = nn.DataParallel(self.disc, device_ids=gpu_ids)
        self.optimizer_d = torch.optim.Adam(self.disc.parameters(), betas=(0.0, args.beta), lr=args.lrd, eps=1e-6)
        self.optimizer_g = torch.optim.Adam(self.gen.parameters(), betas=(0.0, args.beta), lr=args.lrg, eps=1e-6)
        self.optimizers.append(self.optimizer_d)
        self.optimizers.append(self.optimizer_g)
    def trainer(self, data_loader, test_loader, iters=50):
        if args.resume:
            self.load()
        sum_loss_d = 0
        sum_loss_g = 0
        self.writer = SummaryWriter(log_dir=self.log_dir)
        for iteration in range(iters):
            epoch = (iteration) // len(data_loader)+self.epoch
            e_costs = []
            g_costs = []
            #energy function
            for name, param in self.disc.named_parameters():
                param.requires_grad = True
            for name, param in self.gen.named_parameters():
                param.requires_grad = False
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
                self.optimizer_d.zero_grad()
                D_loss.backward()
                self.optimizer_d.step()
            d_real_mean, d_fake_mean,  D_loss_mean, gp_loss_mean= np.mean(e_costs[-args.energy_model_iters:], 0)
            sum_loss_d += D_loss_mean.item()

            # generator
            for name, param in self.disc.named_parameters():
                param.requires_grad = False
            for name, param in self.gen.named_parameters():
                param.requires_grad = True
            for i in range(args.generator_iters):
                z_train = torch.randn((data.shape[0], self.d)).to(self.device)
                z_train.requires_grad_()
                fake= self.gen(z_train)
                d_fake_g = self.disc(fake)
                H=self.compute_entropy_s(z_train)
                g_loss = d_fake_g.mean()-H.mean()*args.H_weight
                g_costs.append([d_fake_g.mean().item(), g_loss.item(), H.mean().item()])
                self.optimizer_g.zero_grad()
                g_loss.backward()
                self.optimizer_g.step()
            d_fake_g_mean, g_loss_mean, H_mean\
                = np.mean(g_costs[-args.generator_iters:], 0)
            sum_loss_g += g_loss_mean.item()

            if iteration % 250 == 0:
                self.writer.add_scalar('D_loss', D_loss_mean, iteration)
                self.writer.add_scalar('g_loss', g_loss_mean, iteration)
                self.writer.add_scalar('H', H_mean, iteration)
            if iteration % (5*len(data_loader)) == 0:
                with torch.no_grad():
                    self.visualize_results(epoch)
            if iteration % (10*len(data_loader)) == 0:
                    self.eval_metric_score(iteration,test_loader)
            if (iteration) % (len(data_loader)) == 0:
                avg_loss_d = sum_loss_d / (iteration+1)
                avg_loss_g = sum_loss_g / (iteration+1)
                print('(MEAN) ====> Epoch: {} Average loss d: {:.4f}'.format(epoch, avg_loss_d))
                print('(MEAN) ====> Epoch: {} Average loss g: {:.4f}'.format(epoch, avg_loss_g))
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
        self.gen.train()
        return H
    def compute_entropy_acc(self, z):
        self.gen.eval()
        _, J = self.gen(z, True)
        J=J.reshape(self.batch_size,-1,self.d)
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

    def compute_ebmpr(self,dataloader):
        pkl_path_d = '/home/congen/code/geoml_gan/models/cifar10/EBM/128/01/1624620823/epoch099_d.pkl'
        pkl_path_g = '/home/congen/code/geoml_gan/models/cifar10/EBM/128/01/1624620823/epoch099_g.pkl'
        self.disc.load_state_dict(torch.load(pkl_path_d))
        self.gen.load_state_dict(torch.load(pkl_path_g))
        self.disc.eval()
        self.gen.eval()
        n_generate = len(dataloader.dataset)
        num_split, num_run4PR, num_cluster4PR, beta4PR = 1, 10, 20, 8
        is_scores, fid_score=compute_is_and_fid(dataloader,self.gen, self.d, args, device,
                                             n_generate=n_generate,splits=num_split)
        precision, recall, f_beta, f_beta_inv = calculate_f_beta_score(dataloader, self.gen, self.d,
                                n_generate,num_run4PR, num_cluster4PR, beta4PR,device)
        PR_Curve = plot_pr_curve(precision, recall,self.result_dir)

        print('IS',is_scores)
        print('FID',fid_score)
        print('F8',f_beta)
        print('F1/8',f_beta_inv)
    def eval_metric_score(self,iteration,dataloader):
        self.disc.eval()
        self.gen.eval()
        n_generate = len(dataloader.dataset)
        num_split, num_run4PR, num_cluster4PR, beta4PR = 1, 10, 20, 8
        is_scores, fid_score = compute_is_and_fid(dataloader,self.gen, self.d, args, device,
                                               n_generate=n_generate, splits=num_split)
        precision, recall, f_beta, f_beta_inv = calculate_f_beta_score(dataloader, self.gen, self.d,
                                                                       n_generate, num_run4PR, num_cluster4PR, beta4PR,
                                                                       device)
        PR_Curve = plot_pr_curve(precision, recall, self.result_dir)

        self.writer.add_scalar('FID', fid_score, iteration)
        self.writer.add_scalar('IS', is_scores, iteration)
        self.writer.add_scalar('F8', f_beta, iteration)
        self.writer.add_scalar('F1/8', f_beta_inv, iteration)
        self.disc.train()
        self.gen.train()
    def save(self, epoch):
        torch.save(self.disc.state_dict(), os.path.join(self.save_dir, 'epoch%03d' % epoch+'_d.pkl'))
        torch.save(self.gen.state_dict(), os.path.join(self.save_dir, 'epoch%03d' % epoch+'_g.pkl'))
        state = {'epoch': epoch, 'optimizers': []}
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())

        save_path = os.path.join(self.save_dir, 'epoch%03d' % epoch + '.state')

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                torch.save(state, save_path)
            except Exception as e:
                print(f'Save training state error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            print(f'Still cannot save {save_path}. Just ignore it.')
    def load(self):
        pkl_path_d = '/home/congen/code/geoml_gan/models/cifar10/EBM/128/1617651553/epoch029_d.pkl'
        pkl_path_g = '/home/congen/code/geoml_gan/models/cifar10/EBM/128/1617651553/epoch029_g.pkl'
        pkl_path_state = '/home/congen/code/EBM-BB-exp/models/mnist/EBM_BB/032/01/1656494414/epoch060.state'
        self.disc.load_state_dict(torch.load(pkl_path_d))
        self.gen.load_state_dict(torch.load(pkl_path_g))
        resume_optimizers = torch.load(pkl_path_state)['optimizers']
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        self.epoch = torch.load(pkl_path_state)['epoch']

@utils.register_model(name='EBM_BB')
class EBM_BB(nn.Module):
    def __init__(self, args, device='cpu'):
        super(EBM_BB, self).__init__()
        self.device = device
        self.d = args.z_dim
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.model_name = args.EBM_type
        self.sample_z_ = torch.randn((self.batch_size, self.d)).to(self.device)
        self.optimizers = []
        self.epoch=0
        if args.input_size == 32:
            module = __import__('network.DCGAN', fromlist=['something'])
            self.disc = module.EnergyModel()
            self.gen = module.Generator(self.d)
        elif args.input_size == 64:
            module = __import__('network.Resnet', fromlist=['something'])
            self.disc = module.EnergyModel(args)
            self.gen = module.Generator(args, self.d)
        with open("{}/args.txt".format(args.result_dir), 'a') as f:
            print('\n', self.disc, '\n', self.gen, file=f)
        self.to(self.device)

        if args.ngpu > 1:
            gpu_ids = range(args.ngpu)
            self.gen=nn.DataParallel(self.gen,device_ids=gpu_ids)
            self.disc = nn.DataParallel(self.disc, device_ids=gpu_ids)
        self.optimizer_d = torch.optim.Adam(self.disc.parameters(), betas=(0.0, args.beta), lr=args.lrd, eps=1e-6)
        self.optimizer_g = torch.optim.Adam(self.gen.parameters(), betas=(0.0, args.beta), lr=args.lrg, eps=1e-6)
        self.optimizers.append(self.optimizer_d)
        self.optimizers.append(self.optimizer_g)
    def trainer(self, data_loader, test_loader, iters=50):
        if args.resume:
            self.load()
        sum_loss_d = 0
        sum_loss_g = 0
        self.writer = SummaryWriter(log_dir=self.log_dir)
        for iteration in range(iters):
            epoch = (iteration) // len(data_loader)+self.epoch
            e_costs = []
            g_costs = []
            #energy function
            for name, param in self.disc.named_parameters():
                param.requires_grad = True
            for name, param in self.gen.named_parameters():
                param.requires_grad = False
            for i in range(args.energy_model_iters):
                data = data_loader.next()[0].cuda()
                data = data.to(self.device)
                data.requires_grad_()
                z_train = torch.randn((data.shape[0], self.d)).to(self.device)
                G_z = self.gen(z_train)
                G_z.requires_grad_()
                d_real = self.disc(data)
                d_fake = self.disc(G_z)
                gradients = torch.autograd.grad(outputs=d_fake,
                            inputs=G_z, grad_outputs=torch.ones_like(d_fake),
                                    allow_unused=True, create_graph=True)[0]
                gradients = gradients.flatten(start_dim=1)
                gp, Jv, H= self.compute_gp(z_train)
                gp_loss = (((gradients * Jv.detach()).sum(-1) - gp.detach()) ** 2).mean()*0.5
                if args.ada == True:
                    gp_weight = 0.7 * (2307 + iteration) ** (-0.55)
                    D_loss = (d_real - d_fake).mean() + (F.relu(gp_weight*gp_loss- args.thre))
                else:
                    D_loss = (d_real - d_fake).mean() + (F.relu(args.gp_weight*gp_loss- args.thre))

                e_costs.append([d_real.mean().item(), d_fake.mean().item(),D_loss.item(), gp_loss.item()])
                self.optimizer_d.zero_grad()
                D_loss.backward()
                self.optimizer_d.step()
            d_real_mean, d_fake_mean,  D_loss_mean, gp_loss_mean= np.mean(e_costs[-args.energy_model_iters:], 0)
            sum_loss_d += D_loss_mean.item()

            # generator
            for name, param in self.disc.named_parameters():
                param.requires_grad = False
            for name, param in self.gen.named_parameters():
                param.requires_grad = True
            for i in range(args.generator_iters):
                z_train2 = torch.randn((data.shape[0], self.d)).to(self.device)
                z_train2.requires_grad_()
                fake= self.gen(z_train2)
                d_fake_g = self.disc(fake)
                g_loss = d_fake_g.mean()-H*args.H_weight
                g_costs.append([d_fake_g.mean().item(), g_loss.item(), H.item()])
                self.optimizer_g.zero_grad()
                g_loss.backward()
                self.optimizer_g.step()
            d_fake_g_mean, g_loss_mean, H_mean= np.mean(g_costs[-args.generator_iters:], 0)
            sum_loss_g += g_loss_mean.item()

            if iteration % 250 == 0:
                self.writer.add_scalar('D_loss', D_loss_mean, iteration)
                self.writer.add_scalar('g_loss', g_loss_mean, iteration)
                self.writer.add_scalar('H', H_mean, iteration)
            if iteration % (5*len(data_loader)) == 0:
                with torch.no_grad():
                    self.visualize_results(epoch)
            if (iteration+1) % (10*len(data_loader)) == 0:
                    self.eval_metric_score(iteration,test_loader)
            if (iteration) % (len(data_loader)) == 0:
                avg_loss_d = sum_loss_d / (iteration+1)
                avg_loss_g = sum_loss_g / (iteration+1)
                print('(MEAN) ====> Epoch: {} Average loss d: {:.4f}'.format(epoch, avg_loss_d))
                print('(MEAN) ====> Epoch: {} Average loss g: {:.4f}'.format(epoch, avg_loss_g))
            if (iteration+1) % (20*len(data_loader)) == 0:
                self.save(epoch)

    def RaRitz(self,v, r, p, intermediate, projection, device):
        JV = []
        r = r / r.mean(-1, True)
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
        v_min = torch.svd(JV).V[:, :, -1:].to(device)
        p_op = V[:, :, -2:].bmm(v_min[:, -2:]).squeeze(-1)
        p_op_norm = p_op.norm(2, dim=-1)
        p.data.index_copy_(0, torch.where(~p_op_norm.isnan())[0], p_op[~p_op_norm.isnan()].detach())
        v_op = V.bmm(v_min).squeeze(-1)
        v_op_norm = v_op.norm(2, dim=-1)
        v.data.index_copy_(0, torch.where(~v_op_norm.isnan())[0], v_op[~v_op_norm.isnan()].detach())
        return v, p

    def compute_gp(self,z):
        z.requires_grad_()
        self.gen.eval()
        logpz = td.Normal(loc=torch.zeros(z.shape[-1]).to(device),
                          scale=torch.ones(z.shape[-1]).to(device)).log_prob(z).sum(1)
        fake_eval = self.gen(z)
        v0 = torch.randn(z.shape).to(device)
        v = v0.clone()
        p = torch.randn(z.shape).to(device)
        projection = torch.ones_like(fake_eval, requires_grad=True).to(device)
        intermediate = torch.autograd.grad(fake_eval, z, projection, create_graph=True)
        Jv = torch.autograd.grad(intermediate[0], projection, v, retain_graph=True)[0]
        size = len(Jv.shape)

        mu = Jv.norm(2, dim=list(range(1, size))) / v.norm(2, dim=-1)
        for i in range(args.ssv_iter):
            JTJv = (Jv.detach() * fake_eval).sum(list(range(1, size)), True)
            r = torch.autograd.grad(JTJv, z, torch.ones_like(JTJv, requires_grad=True), retain_graph=True)[0] \
                - (mu ** 2).unsqueeze(-1) * v
            v, p = self.RaRitz(v, r, p, intermediate[0], projection, device)

            Jv = torch.autograd.grad(intermediate[0], projection, v.detach(), create_graph=True)[0]
            mu = Jv.norm(2, dim=list(range(1, size))) / (v.norm(2, dim=-1))

        est = z.shape[-1] * torch.log(mu)
        logpGz = logpz - est
        deri = torch.autograd.grad(-logpGz, z, torch.ones_like(logpGz, requires_grad=True), retain_graph=True)[0]
        #gp = (deri * v0/v0.norm(2,dim=-1,keepdim=True)).sum(-1)
        gp = (deri * v0).sum(-1)
        Jv0 = torch.autograd.grad(intermediate[0], projection, v0.detach(), retain_graph=True)[0].flatten(start_dim=1)
        #Jv0 = Jv0 / v0.norm(2, dim=-1, keepdim=True)
        self.gen.train()
        return gp, Jv0, est[torch.where(~est.isnan()) and torch.where(~est.isinf())].mean()
    def compute_gp_acc(self,z):
        z.requires_grad_()
        self.gen.eval()
        logpz = td.Normal(loc=torch.zeros(z.shape[-1]).to(device),
                          scale=torch.ones(z.shape[-1]).to(device)).log_prob(z).sum(1)
        fake_eval,J = self.gen(z,True)
        J = J.reshape(self.batch_size, -1, self.d)
        v0 = torch.randn(z.shape).to(device)
        jtj = torch.bmm(torch.transpose(J, -2, -1), J)
        H = 0.5 * torch.slogdet(jtj)[1]
        logpGz = logpz - H
        deri = torch.autograd.grad(-logpGz, z, torch.ones_like(logpGz, requires_grad=True), allow_unused=True,
                                   create_graph=True)[0]
        gp = (deri * v0).sum(-1)
        Jv0 = J.bmm(v0.unsqueeze(-1)).squeeze(-1)
        self.gen.train()
        return gp, Jv0, H.mean()
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
        self.gen.train()
        return H
    def compute_entropy_acc(self, z):
        self.gen.eval()
        _, J = self.gen(z, True)
        J=J.reshape(self.batch_size,-1,self.d)
        jtj = torch.bmm(torch.transpose(J, -2, -1), J)
        H = 0.5 * torch.slogdet(jtj)[1]
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
    def compute_ebmpr(self,dataloader):
        pkl_path_d = '/home/congen/code/geoml_gan/models/cifar10/EBM/128/01/1624620823/epoch099_d.pkl'
        pkl_path_g = '/home/congen/code/geoml_gan/models/cifar10/EBM/128/01/1624620823/epoch099_g.pkl'
        self.disc.load_state_dict(torch.load(pkl_path_d))
        self.gen.load_state_dict(torch.load(pkl_path_g))
        self.disc.eval()
        self.gen.eval()
        n_generate = len(dataloader.dataset)
        num_split, num_run4PR, num_cluster4PR, beta4PR = 1, 10, 20, 8
        is_scores, fid_score=compute_is_and_fid(dataloader,self.gen, self.d, args, device,
                                                 n_generate=n_generate,splits=num_split)
        precision, recall, f_beta, f_beta_inv = calculate_f_beta_score(dataloader, self.gen, self.d,
                                n_generate,num_run4PR, num_cluster4PR, beta4PR,device)
        PR_Curve = plot_pr_curve(precision, recall,self.result_dir)

        print('IS',is_scores)
        print('FID',fid_score)
        print('F8',f_beta)
        print('F1/8',f_beta_inv)
    def eval_metric_score(self,iteration,dataloader):
        self.disc.eval()
        self.gen.eval()
        #fid_cache = '/home/congen/code/AGE/data/tf_fid_stats_cifar10_32.npz'
        n_generate = len(dataloader.dataset)
        num_split, num_run4PR, num_cluster4PR, beta4PR = 1, 10, 20, 8
        is_scores, fid_score = compute_is_and_fid(dataloader,self.gen, self.d, args, device,
                                                   n_generate=n_generate, splits=num_split)
        precision, recall, f_beta, f_beta_inv = calculate_f_beta_score(dataloader, self.gen, self.d,
                                                                       n_generate, num_run4PR, num_cluster4PR, beta4PR,
                                                                       device)
        PR_Curve = plot_pr_curve(precision, recall, self.result_dir)

        self.writer.add_scalar('FID', fid_score, iteration)
        self.writer.add_scalar('IS', is_scores, iteration)
        self.writer.add_scalar('F8', f_beta, iteration)
        self.writer.add_scalar('F1/8', f_beta_inv, iteration)
        self.disc.train()
        self.gen.train()
    def save(self, epoch):
        torch.save(self.disc.state_dict(), os.path.join(self.save_dir, 'epoch%03d' % epoch+'_d.pkl'))
        torch.save(self.gen.state_dict(), os.path.join(self.save_dir, 'epoch%03d' % epoch+'_g.pkl'))
        state = {'epoch': epoch, 'optimizers': []}
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())

        save_path = os.path.join(self.save_dir, 'epoch%03d' % epoch + '.state')

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                torch.save(state, save_path)
            except Exception as e:
                print(f'Save training state error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            print(f'Still cannot save {save_path}. Just ignore it.')

    def load(self):
        pkl_path_d = '/home/congen/code/EBM-BB/models/cifar10/EBM_BB/128/01/1641761825/epoch099_d.pkl'
        pkl_path_g = '/home/congen/code/EBM-BB/models/cifar10/EBM_BB/128/01/1641761825/epoch099_g.pkl'
        pkl_path_state = '/home/congen/code/EBM-BB-exp/models/mnist/EBM_BB/032/01/1656494414/epoch060.state'
        self.disc.load_state_dict(torch.load(pkl_path_d))
        self.gen.load_state_dict(torch.load(pkl_path_g))
        resume_optimizers = torch.load(pkl_path_state)['optimizers']
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        self.epoch = torch.load(pkl_path_state)['epoch']




if __name__ == "__main__":
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=2
        export PORT=6006
        export CUDA_HOME=/opt/cuda/cuda-11.0
        export TIME_STR=1
        export PYTHONPATH=./
        python 	./train/train_image.py


    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    desc = "Pytorch implementation of EBM collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--EBM_type', type=str, default='EBM_BB',
                        choices=['EBM_BB', 'EBM_0GP'], help='The type of EBM')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10','animeface'],
                        help='The name of dataset')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train',  'ebmpr'],help='mode')
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--bn", type=bool, default=False)
    parser.add_argument("--sn", type=bool, default=False)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument('--epoch', type=int, default=260, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=128, help='The size of latent space')
    parser.add_argument('--input_size', type=int, default=32, help='The size of input image')
    parser.add_argument("--lrd", type=float, default=5e-5)
    parser.add_argument("--lrg", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.999)
    parser.add_argument("--ada", type=bool, default=False)
    parser.add_argument("--thre", type=float, default=0)
    parser.add_argument('--energy_model_iters', type=int, default=1)
    parser.add_argument("--generator_iters", type=int, default=1)
    parser.add_argument("--ssv_iter", type=int, default=2)
    parser.add_argument("--gp_weight", type=float, default=0.001)
    parser.add_argument("--H_weight", type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the network')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--benchmark_mode', type=bool, default=True)

    args = parser.parse_args()
    # --save_dir
    args.seed = utils.setup_seed(args.seed)
    if args.mode == 'train' and utils.is_debugging() == False:
        time = int(time.time())
        args.save_dir = os.path.join(args.save_dir + '/' + args.dataset + '/'
                + args.EBM_type + '/%03d' % args.z_dim +'/%02d' % args.energy_model_iters+ '/%03d' % time)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # --result_dir
        args.result_dir = os.path.join(args.result_dir + '/' + args.dataset + '/'
            + args.EBM_type + '/%03d' % args.z_dim +'/%02d' % args.energy_model_iters+ '/%03d' % time)
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)

        # --log_dir
        args.log_dir = os.path.join(args.log_dir + '/' + args.dataset + '/'
                + args.EBM_type + '/%03d' % args.z_dim + '/%02d' % args.energy_model_iters+'/tb_%03d' % time)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        with open("{}/args.txt".format(args.result_dir), 'w') as f:
            json.dump(args.__dict__, f, indent=4, sort_keys=True)
    elif utils.is_debugging() == True:
        # --save_dir
        args.save_dir = os.path.join(args.save_dir + '/debug')
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # --result_dir
        args.result_dir = os.path.join(args.result_dir + '/debug')
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)

        # --log_dir
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

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dataset_train = data.LoadDataset(args,train=True)
    dataset_test = data.LoadDataset(args,train=False)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_loader = dict(train=data.InfiniteDataLoader(train_loader))
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False)
    # Fit network
    model= utils.get_model(args.EBM_type)(args, device)
    if args.mode == 'train':
        iters = args.epoch * len(train_loader['train'])+1
        model.trainer(train_loader['train'],test_loader, iters)
    elif args.mode == 'ebmpr':
        model.compute_ebmpr(test_loader)




