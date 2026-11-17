"""PLFCL: Part-Aware Local Feature Contrastive Learning for 3D Hand Pose Estimation."""

import torch.nn as nn
import math
import numpy as np
import torch
from model.mesh2joint_contrastive import mesh2joint_intra_contrastive_loss, mesh2joint_inter_contrastive_loss
from model.transformer import Transformer
from pointutil import Conv1d, PointNetSetAbstraction, BiasConv1d, square_distance, index_points_group
import torch.nn.functional as F
from convNeXT.resnetUnet import convNeXTUnetBig
from utils import vis_3d_skeleton


def smooth_l1_loss(input, target, sigma=10., reduce=True, normalizer=1.0):
    beta = 1. / (sigma ** 2)
    diff = torch.abs(input - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    if reduce:
        return torch.sum(loss) / normalizer
    return torch.sum(loss, dim=1) / normalizer
criterion = smooth_l1_loss

model_list = {
          'tiny': ([3, 3, 9, 3], [96, 192, 384, 768]),
          'small': ([3, 3, 27, 3], [96, 192, 384, 768]),
          'base': ([3, 3, 27, 3], [128, 256, 512, 1024]),
          'large': ([3, 3, 27, 3], [192, 384, 768, 1536])
          }
weight_url_1k = {
    'tiny': "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224.pth",
    'small': "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224.pth",
    'base': "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224.pth",
    'large': "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224.pth"
}

weight_url_22k = {
    'tiny': "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    'small': "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    'base': "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    'large': "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth"
}



class LocalFoldingGraph(nn.Module):
    def __init__(self, nsample, in_channel, latent_channel, mlp, njoint, mlp2=None, bn = False, use_leaky = True,  radius=False, residual=True):
        super(LocalFoldingGraph,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.residual = residual
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp2 = mlp2
        self.graphs_a = nn.ParameterList()
        self.graphs_w = nn.ModuleList()
        
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3
        last_latent_channel = latent_channel
        for out_channel in mlp:
            if torch.cuda.is_available():
                self.graphs_a.append(nn.Parameter(torch.randn(1, last_latent_channel, njoint, njoint).cuda(),requires_grad=True))
            else:
                self.graphs_a.append(
                    nn.Parameter(torch.randn(1, last_latent_channel, njoint, njoint), requires_grad=True))
            self.graphs_w.append(Conv1d(last_latent_channel, last_latent_channel, 1))
            self.mlp_convs.append(nn.Conv2d(last_channel + last_latent_channel + 3 + 3, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
            last_latent_channel = out_channel
        if mlp2:
            self.mlp2_convs = nn.ModuleList()
            if bn:
                self.mlp2_bns = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Conv1d(last_channel, out_channel, 1, bias=False))
                if bn:
                    self.mlp2_bns.append(nn.BatchNorm1d(out_channel))

                last_channel = out_channel

        self.regress = nn.Conv1d(in_channels=last_channel, out_channels=3, kernel_size=1)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(0.1, inplace=True)

    def forward(self, joints, xyz, points, beta, joint_idx, latents):
        '''
        joints: joints [B, 3, J]
        xyz: local points [B, 3, N]
        points1: joints features [B, C, J]
        points: local features [B, C, N]
        '''
        B, C, J = joints.shape
        # print(joints.shape)
        _, _, N = xyz.shape
        _, D1, _ = latents.shape
        # _, C, _ = points.shape
        joints = joints.permute(0, 2, 1)
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        beta = beta.view(B, 1)          # (B, 1)
        joint_idx = joint_idx.unsqueeze(0).repeat(B,1).unsqueeze(1) # (B, 1, J)
        joint_emb = torch.cat([joint_idx, torch.sin(joint_idx), torch.cos(joint_idx)], dim=1) # (B, 3, J)
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 3)
        time_emb = time_emb.view(B, 3, 1, 1).repeat(1, 1, self.nsample, J)
        # print(grouped_context.shape, joint_emb.shape)

        # local features
        sqrdists = square_distance(joints, xyz)
        dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
        neighbor_xyz = index_points_group(xyz, knn_idx)
        direction_xyz = neighbor_xyz - joints.view(B, J, 1, C)
        grouped_points = index_points_group(points, knn_idx) # B, J, nsample, D2
        new_points = torch.cat([grouped_points, direction_xyz], dim = -1).permute(0, 3, 2, 1)  # [B, D1+D2+3, nsample, J]

        for i, conv in enumerate(self.mlp_convs):
            latents = self.graphs_w[i](torch.matmul(latents.unsqueeze(-2),self.graphs_a[i]).squeeze(-2))
            grouped_context = torch.cat((time_emb, latents.unsqueeze(2).repeat(1,1,self.nsample,1), joint_emb.unsqueeze(2).repeat(1,1,self.nsample,1)),1)
            new_points = torch.cat((grouped_context, new_points), 1)
            if self.bn:
                bn = self.mlp_bns[i]
                new_points = self.relu(bn(conv(new_points)))
            else:
                new_points = self.relu(conv(new_points))
            latents = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        if self.mlp2:
            for i, conv in enumerate(self.mlp2_convs):
                if self.bn:
                    bn = self.mlp2_bns[i]
                    new_points =  self.relu(bn(conv(new_points)))
                else:
                    new_points =  self.relu(conv(new_points))
                    # new_points =  self.relu(self.drop(conv(new_points)))

        joint_res = self.regress(new_points)

        if self.residual:
            joint_res = joint_res + joints.transpose(1,2)

        return joint_res, new_points


class JointRefinementGraph(nn.Module):
    def __init__(self, nsample, in_channel, latent_channel, mlp, njoint, mlp2=None, bn=False, use_leaky=True,
                 radius=False, residual=True):
        super(JointRefinementGraph, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.residual = residual
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp2 = mlp2
        self.graphs_a = nn.ParameterList()
        self.graphs_w = nn.ModuleList()

        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3
        last_latent_channel = latent_channel
        for out_channel in mlp:
            if torch.cuda.is_available():
                self.graphs_a.append(
                    nn.Parameter(torch.randn(1, last_latent_channel, njoint, njoint).cuda(), requires_grad=True))
            else:
                self.graphs_a.append(
                    nn.Parameter(torch.randn(1, last_latent_channel, njoint, njoint), requires_grad=True))
            self.graphs_w.append(Conv1d(last_latent_channel, last_latent_channel, 1))
            self.mlp_convs.append(nn.Conv2d(last_channel + last_latent_channel + 3, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
            last_latent_channel = out_channel
        if mlp2:
            self.mlp2_convs = nn.ModuleList()
            if bn:
                self.mlp2_bns = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Conv1d(last_channel, out_channel, 1, bias=False))
                if bn:
                    self.mlp2_bns.append(nn.BatchNorm1d(out_channel))

                last_channel = out_channel

        self.regress = nn.Conv1d(in_channels=last_channel, out_channels=3, kernel_size=1)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(0.1, inplace=True)

    def forward(self, joints, xyz, points, joint_idx, latents):
        '''
        joints: joints [B, 3, J]
        xyz: local points [B, 3, N]
        points1: joints features [B, C, J]
        points: local features [B, C, N]
        '''
        B, C, J = joints.shape
        # print(joints.shape)
        _, _, N = xyz.shape
        _, D1, _ = latents.shape
        # _, C, _ = points.shape
        joints = joints.permute(0, 2, 1)
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        joint_idx = joint_idx.unsqueeze(0).repeat(B, 1).unsqueeze(1)  # (B, 1, J)
        joint_emb = torch.cat([joint_idx, torch.sin(joint_idx), torch.cos(joint_idx)], dim=1)  # (B, 3, J)

        # local features
        sqrdists = square_distance(joints, xyz)
        dists, knn_idx = torch.topk(sqrdists, self.nsample, dim=-1, largest=False, sorted=False)
        neighbor_xyz = index_points_group(xyz, knn_idx)
        direction_xyz = neighbor_xyz - joints.view(B, J, 1, C)
        grouped_points = index_points_group(points, knn_idx)  # B, J, nsample, D2
        new_points = torch.cat([grouped_points, direction_xyz], dim=-1).permute(0, 3, 2, 1)  # [B, D1+D2+3, nsample, J]

        for i, conv in enumerate(self.mlp_convs):
            latents = self.graphs_w[i](torch.matmul(latents.unsqueeze(-2), self.graphs_a[i]).squeeze(-2))
            grouped_context = torch.cat((latents.unsqueeze(2).repeat(1, 1, self.nsample, 1),
                                         joint_emb.unsqueeze(2).repeat(1, 1, self.nsample, 1)), 1)
            new_points = torch.cat((grouped_context, new_points), 1)
            if self.bn:
                bn = self.mlp_bns[i]
                new_points = self.relu(bn(conv(new_points)))
            else:
                new_points = self.relu(conv(new_points))
            latents = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        if self.mlp2:
            for i, conv in enumerate(self.mlp2_convs):
                if self.bn:
                    bn = self.mlp2_bns[i]
                    new_points = self.relu(bn(conv(new_points)))
                else:
                    new_points = self.relu(conv(new_points))
                    # new_points =  self.relu(self.drop(conv(new_points)))

        joint_res = self.regress(new_points)

        if self.residual:
            joint_res = joint_res + joints.transpose(1, 2)

        return joint_res, new_points

class LocalFolding(nn.Module):
    def __init__(self, nsample, in_channel, ctx_channel, mlp, njoint, mlp2=None, bn = False, use_leaky = True,  radius=False, residual=True):
        super(LocalFolding,self).__init__()

        self.mappling_graph1 = LocalFoldingGraph(nsample=nsample, in_channel=in_channel, latent_channel=ctx_channel, mlp=mlp, njoint=njoint, mlp2=mlp2, bn=bn, use_leaky=use_leaky, radius=radius, residual=residual)
        self.mappling_graph2 = LocalFoldingGraph(nsample=nsample, in_channel=in_channel//2, latent_channel=mlp[-1], mlp=mlp, njoint=njoint, mlp2=mlp2, bn=bn, use_leaky=use_leaky, radius=radius, residual=residual)
        self.mappling_graph3 = LocalFoldingGraph(nsample=nsample, in_channel=in_channel, latent_channel=ctx_channel, mlp=mlp, njoint=njoint, mlp2=mlp2, bn=bn, use_leaky=use_leaky, radius=radius, residual=residual)
        self.mappling_graph4 = LocalFoldingGraph(nsample=nsample, in_channel=in_channel//2, latent_channel=mlp[-1], mlp=mlp, njoint=njoint, mlp2=mlp2, bn=bn, use_leaky=use_leaky, radius=radius, residual=residual)

    def forward(self, joints, xyz, points, beta, joint_idx, latents, img, img_feat):
        
        img_feat = img_feat.view(img_feat.size(0), img_feat.size(1), -1)
        joints0, latents = self.mappling_graph1(joints, xyz, points, beta, joint_idx, latents)
        # print(latents.shape)
        joints1, latents = self.mappling_graph2(joints0, img.transpose(1,2).contiguous(), img_feat, beta, joint_idx, latents)

        joints2, latents = self.mappling_graph3(joints1, xyz, points, beta, joint_idx, latents)
        joints3, latents = self.mappling_graph4(joints2, img.transpose(1,2).contiguous(), img_feat, beta, joint_idx, latents)

        return joints1, joints3


class JointRefinement(nn.Module):
    def __init__(self, nsample, in_channel, ctx_channel, mlp, njoint, mlp2=None, bn=False, use_leaky=True, radius=False,
                 residual=True):
        super(JointRefinement, self).__init__()

        self.mappling_graph1 = JointRefinementGraph(nsample=nsample, in_channel=in_channel, latent_channel=ctx_channel,
                                                 mlp=mlp, njoint=njoint, mlp2=mlp2, bn=bn, use_leaky=use_leaky,
                                                 radius=radius, residual=residual)
        self.mappling_graph2 = JointRefinementGraph(nsample=nsample, in_channel=in_channel // 2, latent_channel=mlp[-1],
                                                 mlp=mlp, njoint=njoint, mlp2=mlp2, bn=bn, use_leaky=use_leaky,
                                                 radius=radius, residual=residual)
        self.mappling_graph3 = JointRefinementGraph(nsample=nsample, in_channel=in_channel, latent_channel=ctx_channel,
                                                 mlp=mlp, njoint=njoint, mlp2=mlp2, bn=bn, use_leaky=use_leaky,
                                                 radius=radius, residual=residual)
        self.mappling_graph4 = JointRefinementGraph(nsample=nsample, in_channel=in_channel // 2, latent_channel=mlp[-1],
                                                 mlp=mlp, njoint=njoint, mlp2=mlp2, bn=bn, use_leaky=use_leaky,
                                                 radius=radius, residual=residual)

        self.criterion = nn.CosineSimilarity(dim=1).cuda()
        self.predictor0 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1)
        )
        # self.predictor1 = nn.Sequential(
        #     nn.Conv1d(512, 128, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(128, 64, 1)
        # )

    def forward(self, joints, xyz, points, joint_idx, latents, img, img_feat):
        img_feat = img_feat.view(img_feat.size(0), img_feat.size(1), -1)
        joints0, latents0 = self.mappling_graph1(joints, xyz, points, joint_idx, latents)
        # print(latents.shape)
        joints1, latents = self.mappling_graph2(joints0, img.transpose(1, 2).contiguous(), img_feat,  joint_idx,
                                                latents0)
        # print(latents.shape)
        # loss0 = mesh2joint_intra_contrastive_loss(latents0, latents, joints0, joints1, torch.ones_like(joints0.permute(0, 2, 1)).squeeze(-1),
        #                            anchour_num=100, pos_num=64, neg_num=512, temperature=0.07)

        # loss = mesh2joint_intra_contrastive_loss(latents, latents0, joints1, joints0,
        #                            torch.ones_like(joints0.permute(0, 2, 1)).squeeze(-1),
        #                            anchour_num=100, pos_num=64, neg_num=512, temperature=0.02)

        # loss = torch.mean(torch.stack([loss0, loss]), 0)

        joints2, latents2 = self.mappling_graph3(joints1, xyz, points,  joint_idx, latents)
        joints3, latents3 = self.mappling_graph4(joints2, img.transpose(1, 2).contiguous(), img_feat,  joint_idx,
                                                latents)

        # loss = mesh2joint_intra_contrastive_loss(latents2, latents3, joints2, joints3,
        #                             torch.ones_like(joints0.permute(0, 2, 1)).squeeze(-1),
        #                             anchour_num=100, pos_num=64, neg_num=512, temperature=0.07)

        loss = mesh2joint_intra_contrastive_loss(latents3, latents2, joints3, joints2,
                                   torch.ones_like(joints0.permute(0, 2, 1)).squeeze(-1),
                                   anchour_num=100, pos_num=64, neg_num=512, temperature=0.07)

        return joints1, joints3, loss


class VarianceSchedule(nn.Module):

    def __init__(self, num_steps, mode='cosine'):
        super().__init__()
        assert mode in ('linear', "cosine", "real_linear")
        self.num_steps = num_steps

        self.mode = mode

        t = torch.linspace(0, 1, steps=num_steps+1)
        if mode == 'linear':
            self.log_snr = self.beta_linear_log_snr
        elif mode == "cosine":
            self.log_snr = self.alpha_cosine_log_snr


    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(0, 1), batch_size)
        return ts.tolist()

    def log_snr_to_alpha_sigma(self, log_snr):
        return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

    def beta_linear_log_snr(self, t):
        return -torch.log(torch.special.expm1(1e-4 + 10 * (t ** 2)))

    def alpha_cosine_log_snr(self, t, s: float = 0.008):
        return -torch.log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1)

    def real_linear_beta_schedule(self, timesteps):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = 1 - x / timesteps
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)


class DiffusionPoint(nn.Module):

    def __init__(self, net, joints, var_sched:VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched
        self.joints = joints
        joint_idx = torch.linspace(0, 1, steps=joints)
        self.register_buffer('joint_idx', joint_idx)
        self.eta = 0.

    def get_loss(self, x_0, context, xyz, points, img, img_feat, t=None):
        """
        Args:
            x_0:  Input joint, (B, J, 3).
            context:  Shape latent, (B, J, F).
            xyz:  Input points, (B, N, F).
            points:  Input features, (B, N, F).
        """
        batch_size, point_dim, _ = x_0.size()
        # print(x_0.shape)
        times = torch.zeros(
            (batch_size,), device=x_0.device).float().uniform_(0, 1)
        log_snr = self.var_sched.log_snr(times)
        alpha, sigma = self.var_sched.log_snr_to_alpha_sigma(times)
        c0 = alpha.view(-1, 1, 1)       # (B, 1, 1)
        c1 = sigma.view(-1, 1, 1)   # (B, 1, 1)

        e_rand = torch.randn_like(x_0)  # (B, d, J)
        x_t = c0 * x_0 + c1 * e_rand
        x_0_hat0, x_0_hat1 = self.net(x_t, xyz, points, log_snr, self.joint_idx, context, img, img_feat)

        loss = smooth_l1_loss(x_0_hat0, x_0)+smooth_l1_loss(x_0_hat1, x_0)

        return loss

    def sample(self, context, xyz, points, img, img_feat, sampling_steps=10, point_dim=3, flexibility=0.0, ret_traj=False):
        batch_size = context.size(0)
        num_joints = self.joints

        times = torch.linspace(1, 0, steps=sampling_steps + 1, device=context.device)

        times = times.unsqueeze(0).repeat(batch_size, 1)
        # print(times.shape)

        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        # print(times.shape)

        time_pairs = times.unbind(dim=-1)
        # print(time_pairs.size)

        x_t = torch.randn([batch_size, point_dim, num_joints]).to(context.device)
        for time, time_next in time_pairs:
            log_snr = self.var_sched.log_snr(time)
            _, x_0_hat = self.net(x_t, xyz, points, log_snr, self.joint_idx, context, img, img_feat)
            # x_0_hat, _ = self.net(x_t, xyz, points, log_snr, self.joint_idx, context, img, img_feat)

            log_snr_next = self.var_sched.log_snr(time_next)
            alpha, sigma = self.var_sched.log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = self.var_sched.log_snr_to_alpha_sigma(log_snr_next)

            pred_noise = (x_t - alpha.view(-1, 1, 1) * x_0_hat) / \
                sigma.view(-1, 1, 1).clamp(min=1e-8)
            x_t = x_0_hat * alpha_next.view(-1, 1, 1) + pred_noise * sigma_next.view(-1, 1, 1)
            
        return x_0_hat


    def sample_mh(self, context, xyz, points, img, img_feat, h=10, sampling_steps=20, point_dim=3, flexibility=0.0, ret_traj=False):

        hypothesis_num = h
        batch_size = context.size(0)
        # print(batch_size)
        num_joints = self.joints
        hypothesis = torch.zeros([batch_size, hypothesis_num, point_dim, num_joints]).to(context.device)
        
        for i in range(hypothesis_num):
            x_0_hat = self.sample(context, xyz, points, img, img_feat, sampling_steps=sampling_steps, point_dim=point_dim, flexibility=flexibility, ret_traj=ret_traj)
            hypothesis[:,i] = x_0_hat

        return hypothesis


class Mesh2JointContrastive(nn.Module):
    """Refinement wrapper for mesh-to-joint contrastive learning (Section III-B)."""

    def __init__(self, net, joints):
        super().__init__()
        self.net = net
        self.joints = joints
        joint_idx = torch.linspace(0, 1, steps=joints)
        self.register_buffer('joint_idx', joint_idx)
        self.eta = 0.

    def get_loss(self, x_0, context, xyz, points, img, img_feat, gt):
        """
        Args:
            x_0:  Input joint, (B, J, 3).
            context:  Shape latent, (B, J, F).
            xyz:  Input points, (B, N, F).
            points:  Input features, (B, N, F).
        """
        batch_size, point_dim, _ = x_0.size()
        # print(x_0.shape)
        x_0_hat0, x_0_hat1 , loss = self.net(x_0, xyz, points, self.joint_idx, context, img, img_feat)

        loss = smooth_l1_loss(x_0_hat0, gt) + smooth_l1_loss(x_0_hat1, gt) + 0.001 * loss

        return loss

    def sample(self, x_0, context, xyz, points, img, img_feat):
        x_0_hat, _, _ = self.net(x_0, xyz, points, self.joint_idx, context, img, img_feat)

        return x_0_hat

    def sample_mh(self, x0, context, xyz, points, img, img_feat, h=10, point_dim=3):

        hypothesis_num = h
        batch_size = context.size(0)
        # print(batch_size)
        num_joints = self.joints
        hypothesis = torch.zeros([batch_size, hypothesis_num, point_dim, num_joints]).to(context.device)

        for i in range(hypothesis_num):
            x_0_hat = self.sample(x0, context, xyz, points, img, img_feat)
            hypothesis[:, i] = x_0_hat

        return hypothesis

class PLFCL(nn.Module):
    """Part-Aware Local Feature Contrastive Learning (PLFCL)."""
    def __init__(self, joints=21, iters=10):
        super(PLFCL, self).__init__()
        
        self.backbone = convNeXTUnetBig('small', pretrain='1k', deconv_dim=128)

        self.encoder_1 = PointNetSetAbstraction(npoint=512, radius=0.1, nsample=64, in_channel=3, mlp=[32,32,128])
        
        self.encoder_2 = PointNetSetAbstraction(npoint=128, radius=0.3, nsample=64, in_channel=128, mlp=[64,64,256])

        self.encoder_3 = nn.Sequential(Conv1d(in_channels=256+3, out_channels=128, bn=True, bias=False),
                                       Conv1d(in_channels=128, out_channels=128, bn=True, bias=False),
                                       Conv1d(in_channels=128, out_channels=512, bn=True, bias=False),
                                       nn.MaxPool1d(128,stride=1))

        self.fold1 = nn.Sequential(BiasConv1d(bias_length=joints, in_channels=512+768, out_channels=512, bn=True),
                                    BiasConv1d(bias_length=joints, in_channels=512, out_channels=512, bn=True),
                                    BiasConv1d(bias_length=joints, in_channels=512, out_channels=512, bn=True))

        self.fold2 = Transformer(injection=False)
        self.fold3 = BiasConv1d(bias_length=joints, in_channels=512+768, out_channels=512, bn=True)
        self.regress_1 = nn.Conv1d(in_channels=512, out_channels=3, kernel_size=1)

        self.iters = iters
        self.joints = joints

        self.diffusion = DiffusionPoint(
            net = LocalFolding(nsample=64, in_channel=128+128, ctx_channel=512, mlp=[256, 256, 512, 512], njoint=joints),
            joints=joints,
            var_sched = VarianceSchedule(
                num_steps=iters,
                mode='cosine'
            )
        )

        self.mesh2joint_contrastive = Mesh2JointContrastive(
            net=JointRefinement(nsample=64, in_channel=128+128, ctx_channel=512, mlp=[256, 256, 512, 512], njoint=joints),
            joints=joints,
        )

        self.pc_position_embeddings = nn.Embedding(512, 512)
        self.img_position_embeddings = nn.Embedding(768, 768)

        # self.jointdecoder = nn.Sequential(
        #     nn.Linear(3, 128),
        #     # nn.BatchNorm1d(128*21),
        #     # nn.ReLU(inplace=True),
        #     # nn.Linear(64, 128),
        #     # nn.Linear(128, 128)
        # )
        # self.vit = vit()
        # self.vit.load_state_dict(torch.load('/data/handpose/WorkSpace/PLFCL/mae_pretrain_vit_base.pth'), strict=False)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Migrate the old contrastive module prefix when loading checkpoints."""
        old_prefix = prefix + 'localconstrastive.'
        new_prefix = prefix + 'mesh2joint_contrastive.'
        for key in list(state_dict):
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                if new_key in state_dict:
                    error_msgs.append('Conflicting checkpoint keys: {} and {}'.format(key, new_key))
                else:
                    state_dict[new_key] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def encode(self, pc, feat, img, loader, center, M, cube, cam_para):
        # x: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, y: B*3*sample_num_level1*1

        # _, _, _, c4 = self.backbone.forward_features(img.repeat(1,3,1,1))

        pc1, feat1 = self.encoder_1(pc, feat)# B, 3, 512; B, 64, 512
        
        pc2, feat2 = self.encoder_2(pc1, feat1)# B, 3, 256; B, 128, 256
        
        code = self.encoder_3(torch.cat((pc2, feat2),1))# B, 3, 128; B, 256, 128
        
        pc_img_feat, c4 = self.backbone(img)   # img_offset: B×C×W×H , C=3(direct vector)+1(heatmap)+1(weight)
        # print()
        # # GFexperiment01
        # c4 = self.vit(img)
        img_code = torch.max(c4.view(c4.size(0),c4.size(1),-1),-1,keepdims=True)[0]
        B, C, H, W = pc_img_feat.size()
        img_down = F.interpolate(img, [H, W])
        B, _, N = pc1.size()

        pcl_closeness, pcl_index, img_xyz = loader.img2pcl_index(pc1.transpose(1,2).contiguous(), img_down, center, M, cube, cam_para, select_num=4)

        pcl_feat_index = pcl_index.view(B, 1, -1).repeat(1, C, 1)   # B*128*(K*1024)
        pcl_feat = torch.gather(pc_img_feat.view(B, C, -1), -1, pcl_feat_index).view(B, C, N, -1)

        # # experiment04
        # pcl_feat, _ = torch.max(pcl_feat*pcl_closeness.unsqueeze(1), dim=-1)
        # original
        pcl_feat = torch.sum(pcl_feat*pcl_closeness.unsqueeze(1), dim=-1)
        # # experiment 08
        # pcl_feat = torch.mean(pcl_feat * pcl_closeness.unsqueeze(1), dim=-1)
        feat1 = torch.cat((feat1,pcl_feat),1)

        # original
        code = code.expand(code.size(0),code.size(1), self.joints)
        img_code = img_code.expand(img_code.size(0),img_code.size(1), self.joints)

        # # experiment02
        # input_ids = torch.zeros([code.size(0), self.joints], dtype=torch.long).cuda()
        # position_ids = torch.arange(self.joints, dtype=torch.long, device=input_ids.device)
        # position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        #
        # pc_position_embeddings = self.pc_position_embeddings(position_ids).permute(0, 2, 1)
        # img_position_embeddings = self.img_position_embeddings(position_ids).permute(0, 2, 1)
        #
        # code = pc_position_embeddings + code
        # img_code = img_position_embeddings + img_code

        # # original
        latents = self.fold1(torch.cat((code, img_code), 1))

        # # # GF experiment03
        # codes = torch.cat((img_code, code), 1)
        # latents = self.fold3(codes)
        # latents = self.fold2(latents, latents)
        # latents, joints = self.fold2(torch.cat((code, img_code), 1), torch.cat((code, img_code), 1))
        # # experiment01
        # new_code = torch.cat((code, img_code),1)
        # new_code = new_code.expand(new_code.size(0), new_code.size(1), self.joints)
        # latents = self.fold1(new_code)

        # original
        joints = self.regress_1(latents)

        dense_feat = pc_img_feat.unsqueeze(2)
        loss = mesh2joint_inter_contrastive_loss(dense_feat, joints, anchour_num=100, pos_num=1024, neg_num=2048, temperature=0.07)
        # skeleton = ((0, 1), (0, 5), (0, 9), (0, 13), (0, 17), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11),
        #  (11, 12), (13, 14), (14, 15),
        #  (15, 16), (17, 18), (18, 19), (19, 20))
        # # print(joints[0].shape)
        # vis_3d_skeleton(joints[0].permute(1, 0).cpu().detach().numpy(), torch.ones_like(joints[0].permute(1, 0)), skeleton)
        # loss = mesh2joint_inter_contrastive_loss(dense_feat, joints, anchour_num=100, pos_num=128, neg_num=256, temperature=0.07)
        # return latents, joints, pc1, feat1, img_xyz, pc_img_feat, loss

        return latents, joints, pc1, feat1, img_xyz, dense_feat, 0

    def decode(self, x0, context, xyz, points, img, img_feat, sampling_steps):
        # return self.diffusion.sample_xnoise_mh(context, xyz, points, flexibility=0.0, ret_traj=False)
        # return self.diffusion.sample(context, xyz, points, img, img_feat, sampling_steps=sampling_steps, flexibility=0.0, ret_traj=False)
        return self.mesh2joint_contrastive.sample(x0, context, xyz, points, img, img_feat)
    def forward(self, pc, feat, img, loader, center, M, cube, cam_para,
                sampling_steps=5, operation='predict', gt=None, h=5, step=20):
        if operation == 'loss':
            # One value per sample gives the correct mean for uneven GPU shards.
            return self.get_loss(pc, feat, img, loader, center, M, cube, cam_para, gt).expand(pc.shape[0])
        if operation == 'sample':
            return self.sample(pc, feat, img, loader, center, M, cube, cam_para, h, step)
        if operation != 'predict':
            raise ValueError('Unknown PLFCL operation: {}'.format(operation))
        latents, joints, pc1, feat1, img, img_feat,mesh2joint_contrastive_loss = self.encode(pc, feat, img, loader, center, M, cube, cam_para)
        joints = self.decode(joints, latents, pc1, feat1, img, img_feat, sampling_steps)
        return joints

    def get_loss(self, pc, feat, img, loader, center, M, cube, cam_para, gt):
        latents, joints, pc1, feat1, img, img_feat, mesh2joint_contrastive_loss= self.encode(pc, feat, img, loader, center, M, cube, cam_para)
        # total_loss = self.diffusion.get_loss(gt, latents, pc1, feat1, img, img_feat) + smooth_l1_loss(joints, gt)
        local_loss = self.mesh2joint_contrastive.get_loss(joints, latents, pc1, feat1, img, img_feat, gt)
        # total_loss = total_loss + smooth_l1_loss(joints, gt)
        # experiment 00

        # # 7.06
        # loss = mesh2joint_intra_contrastive_loss(img_feat, gt.permute(0, 2, 1), joints.permute(0, 2, 1), torch.ones_like(joints.permute(0, 2, 1)).squeeze(-1),
        #                            anchour_num=100, pos_num=128, neg_num=256, temperature=0.07)
        # ?
        # print(joints.shape)
        # sample_pred_joint_feat = self.jointdecoder(joints.permute(0, 2, 1))
        # # sample_gt_joint_feat = self.jointdecoder(gt.permute(0, 2, 1))
        # loss = mesh2joint_intra_contrastive_loss(sample_pred_joint_feat,sample_pred_joint_feat, joints.permute(0, 2, 1), joints.permute(0, 2, 1),
        #                            torch.ones_like(joints.permute(0, 2, 1)).squeeze(-1),
        #                            anchour_num=100, pos_num=128, neg_num=256, temperature=0.02)
        # 7.09
        # loss = mesh2joint_intra_contrastive_loss(img_feat, gt.permute(0, 2, 1), joints.permute(0, 2, 1),
        #                            torch.ones_like(joints.permute(0, 2, 1)).squeeze(-1),
        #                            anchour_num=100, pos_num=256, neg_num=512, temperature=0.07)

        # print(loss)

        # loss = mesh2joint_inter_contrastive_loss(sample_pred_joint_feat, joints, anchour_num=100, pos_num=128, neg_num=256, temperature=0.07)
        # return
        total_loss = 0 + smooth_l1_loss(joints, gt) + local_loss
        # total_loss =  local_loss

        return total_loss

    def sample(self, pc, feat, img, loader, center, M, cube, cam_para, h=5, step=20):
        latents, joints, pc1, feat1, img, img_feat = self.encode(pc, feat, img, loader, center, M, cube, cam_para)
        joints = self.diffusion.sample_mh(latents, pc1, feat1, img, img_feat, h, step, flexibility=0.0, ret_traj=False)
        return joints


from thop import profile, clever_format
if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    if torch.cuda.is_available():
        input = torch.randn((2,3,1024)).float().cuda()
        img = torch.randn((2,3,224,224)).float().cuda()
        model = PLFCL().cuda()
    else:
        input = torch.randn((2, 3, 1024)).float()
        img = torch.randn((2, 3, 224, 224)).float()
        model = PLFCL()
    # print(model)

    macs, params = profile(model, inputs=(input,input,img))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.3fM" % (total))
    # model(input, input)
    model.get_loss(input, input, img, input[:,:,:21])

