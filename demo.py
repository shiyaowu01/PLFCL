'''
evaluation
'''
import argparse
import os
import random
import sys
import time
from tqdm import tqdm
import numpy as np
import importlib
from model.parallel import PLFCLDataParallel

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

from mano import MANO
from utils import vis_3d_skeleton

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
parser.add_argument('--main_gpu', type=int, default=0, help='main GPU id')  # CUDA_VISIBLE_DEVICES=0 python eval.py

parser.add_argument('--size', type=str, default='full', help='how many samples do we load: small | full')
parser.add_argument('--bit_width', type=int, default=4, help='quantize for bit width')
parser.add_argument('--SAMPLE_NUM', type=int, default=1024, help='number of sample points')
parser.add_argument('--JOINT_NUM', type=int, default=21, help='number of joints')
parser.add_argument('--INPUT_FEATURE_NUM', type=int, default=3, help='number of input point features')
parser.add_argument('--iters', type=int, default=500, help='start epoch')

parser.add_argument('--save_root_dir', type=str, default='./results', help='output folder')
parser.add_argument('--model', type=str, default='best_model.pth', help='model name for training resume')
parser.add_argument('--test_path', type=str, default='/data/handpose/nyu/dataset/',
                    help='model name for training resume')
parser.add_argument('--protocal', type=str, default='s0', help='model name for training resume')

parser.add_argument('--dataset', type=str, default='nyu', help='optimizer name for training resume')
parser.add_argument('--model_name', type=str, default='plfcl', help='')
parser.add_argument('--gpu', type=str, default='0', help='gpu')

parser.add_argument('--test_step', type=int, default=1, help='number of test timesteps')
parser.add_argument('--test_hypo', type=int, default=2, help='number of test hypotheses')

opt = parser.parse_args()
print(opt)


os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

module = importlib.import_module('network_' + opt.model_name)



opt.manualSeed = 1
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset == 'dexycb':
    save_dir = os.path.join(opt.save_root_dir, opt.dataset + '_' + opt.protocal + '_' + opt.model_name + '_' + str(
        opt.iters) + 'iters' + '_com')
    from dataloader import loader

    opt.JOINT_NUM = 21
elif opt.dataset == 'nyu':
    save_dir = os.path.join(opt.save_root_dir,
                            opt.dataset + '_' + opt.model_name + '_' + str(opt.iters) + 'iters' + '_com')
    from dataloader import loader
    opt.JOINT_NUM = 21

# 1. Load data
if opt.dataset == 'dexycb':
    test_data = loader.DexYCBDataset(opt.protocal, 'test', opt.test_path)
elif opt.dataset == 'nyu':
    test_data = loader.nyu_loader(opt.test_path, 'test', joint_num=opt.JOINT_NUM)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers), pin_memory=False)

print('#Test data:', len(test_data))
print(opt)

# 2. Define model, loss
model = getattr(module, 'PLFCL')(joints=opt.JOINT_NUM, iters=opt.iters)

if torch.cuda.is_available():
    if opt.model != '':
        model.load_state_dict(torch.load(os.path.join(save_dir, opt.model)), strict=False)
    model.cuda()
    criterion = nn.MSELoss(size_average=True).cuda()
    # 3. evaluation
    torch.cuda.synchronize()
else:
    model.load_state_dict(torch.load(os.path.join(save_dir, opt.model), map_location=torch.device('cpu')), strict=False)
    criterion = nn.MSELoss(size_average=True)


# print(model)





if torch.cuda.is_available() and opt.ngpu > 1:
    model = PLFCLDataParallel(model, device_ids=list(range(opt.ngpu)))
model.eval()

h = opt.test_hypo
step = opt.test_step

# print(next(iter(test_dataloader))[0].shape)
with torch.no_grad():
    img, points, gt_xyz, uvd_gt, center, M, cube, cam_para, volume_length= next(iter(test_dataloader))
    # print(gt_xyz.shape)
    vis_joint = torch.ones_like(gt_xyz[4])
    skeleton=((13, 0), (13, 2), (13, 4), (13, 6), (13, 8), (13, 11), (13, 12), (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (9, 10))
    vis_3d_skeleton(uvd_gt[4].clone().detach().cpu().numpy(), vis_joint, skeleton, filename='vis_gt.png')
    print(uvd_gt.shape)
    sys.exit(0)
    volume_length = volume_length.cuda()
    points, gt_xyz, img= points.cuda(), gt_xyz.cuda(), img.cuda()
    center, M, cube, cam_para = center.cuda(), M.cuda(), cube.cuda(), cam_para.cuda()
    estimation= model.sample(points.transpose(1, 2), points.transpose(1, 2), img, test_data, center, M, cube, cam_para,
                              h, step)
    print(estimation.shape)
mano = MANO()

outputs_xyz = torch.zeros_like(estimation).to(estimation.device)
outputs_xyz[:,0] = estimation[:, 0]
for k in range(1, h):
    outputs_xyz[:,k] = (estimation[:, k] + outputs_xyz[:, k-1] * k)/(k+1)
outputs_xyz = estimation.mean(1).transpose(1,2)
vis_joint = torch.ones_like(outputs_xyz[4])
outputs_xyz = outputs_xyz * (cube[2] / 2.0)
gt_xyz = gt_xyz * (cube[2] / 2.0)
# vis_3d_skeleton(outputs_xyz[10].cpu().numpy(), vis_joint, mano.skeleton, filename='vis.png')



# new_joint = mano.get_3d_joints(mesh)
print(gt_xyz[0][0], gt_xyz[2][0])
print(gt_xyz[0][1], gt_xyz[2][1])

vis_3d_skeleton(gt_xyz[2].clone().detach().cpu().numpy(), vis_joint, mano.skeleton, filename='vis_gt.png')
# mesh = mesh.cpu().numpy()*np.array([1,-1,-1])
# save_obj(mesh[4].clone().cpu().numpy(), mano.face, 'output_demo.obj')

# vis_depth(img[4])


# outputs_xyz = torch.zeros_like(joint).to(joint.device)
# outputs_xyz[:, 0] = joint[:, 0]
# for k in range(1, h):
#     outputs_xyz[:, k] = (joint[:, k] + outputs_xyz[:, k - 1] * k) / (k + 1)
#
# outputs_xyz = joint.mean(1).transpose(1, 2)
# mano = MANO()
# vis_joint = torch.ones_like(outputs_xyz[0])
# vis_3d_skeleton(outputs_xyz[0].clone().detach().cpu().numpy(), vis_joint, mano.skeleton, filename='vis.png')

# mesh_sampler = Mesh(num_downsampling = 2, device=torch.device('cuda:0'))
# mesh_sampler1 = Mesh(num_downsampling = 1, device=torch.device('cuda:0'))
#
# mesh_xyz = mesh_sampler.upsample(mesh_xyz)
# mesh_xyz = mesh_sampler1.upsample(mesh_xyz)
# mesh_xyz *= 1000
# # mesh = mesh.cpu().numpy()*np.array([1,-1,-1])
# save_obj(mesh_xyz[0].cpu().numpy()*np.array([1,-1,-1]), mano.face, 'output_demo.obj')

# mesh = mesh_sampler.upsample(mesh.permute(0, 2, 1))
# mesh = mesh_sampler1.upsample(mesh)
# mesh *= 1000
# mesh_joint = mano.get_3d_joints(mesh)
# save_obj(mesh[0].cpu().numpy()*np.array([1,-1,-1]), mano.face, 'output_mesh.obj')
# vis_3d_skeleton(mesh_joint[0].clone().detach().cpu().numpy(), vis_joint, mano.skeleton, filename='mesh_joint.png')