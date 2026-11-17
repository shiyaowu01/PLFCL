'''
utils
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def offset_cal(points,opt):
	#make offset using knn and ball query
	#points: B*(21+1024)*3
        cur_train_size=len(points)
        inputs1_diff = points.transpose(1,2).unsqueeze(1).expand(cur_train_size,opt.JOINT_NUM,3,points.size(1)) \
        			 - points[:,0:opt.JOINT_NUM,:].unsqueeze(-1).expand(cur_train_size,opt.JOINT_NUM,3,points.size(1)) #B*21*3*1045
        inputs1_diff = inputs1_diff[:,:,:,opt.JOINT_NUM:points.size(1)] #B*21*3*1024
        inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)      # B * 21 * 3 * 1024
        inputs1_diff = inputs1_diff.sum(2)                      # B * 21* 1024
        dists, inputs1_idx = torch.topk(inputs1_diff, opt.knn_K, 2, largest=False, sorted=False)  # dists: B * 21 * 64; inputs1_idx: B * 21 * 64

        # ball query
        invalid_map = dists.gt(opt.ball_radius) # B * 21 * 64
        for jj in range(opt.JOINT_NUM):
                inputs1_idx[:,jj,:][invalid_map[:,jj,:]] = jj
        idx_group_l1_long = inputs1_idx.view(cur_train_size,opt.JOINT_NUM*opt.knn_K,1).expand(cur_train_size,opt.JOINT_NUM*opt.knn_K,3)  # B*(21*64)*3
        idx_group_l1_long_long = inputs1_idx.view(cur_train_size,opt.JOINT_NUM,opt.knn_K).unsqueeze(3).expand(cur_train_size,opt.JOINT_NUM,opt.knn_K,4)  # B*21*64*4
        
		
        
		
        #all_idx_long_1 = torch.arange(opt.SAMPLE_NUM, dtype=torch.long).repeat(opt.JOINT_NUM).unsqueeze(0).unsqueeze(2).expand(cur_train_size,opt.JOINT_NUM*opt.SAMPLE_NUM,3) # B*(21*1024)*3
        #all_idx_long_2 = torch.arange(opt.SAMPLE_NUM, dtype=torch.long).repeat(opt.JOINT_NUM) #(21*1024)
        point_cld = points[:,opt.JOINT_NUM:points.size(1),:] 	#points: B*(1024)*3
		
        #point_cld_mlt = point_cld[:,all_idx_long_2,:].view(cur_train_size,opt.JOINT_NUM,opt.SAMPLE_NUM,3)      #B*21*1024*3
        #zeros_1 = torch.zeros(cur_train_size,opt.JOINT_NUM*opt.SAMPLE_NUM,3)   #B*(21*1024)*3
        #zeros_1[idx_group_l1_long] = point_cld[:,:,:].gather(1,idx_group_l1_long)  # zeros_1=B*(21*1024)*3
        inputs_level1 = point_cld[:,:,:].gather(1,idx_group_l1_long).view(cur_train_size,opt.JOINT_NUM,opt.knn_K,3) # B*21*64*3
        inputs_level1_center = points[:,0:opt.JOINT_NUM,:].unsqueeze(2)       # B*21*1*3
        offset = inputs_level1[:,:,:,:] - inputs_level1_center.expand(cur_train_size,opt.JOINT_NUM,opt.knn_K,3) # B*21*64*3
        dist =  torch.mul(offset, offset) # B*21*64*3
        dist = offset.sum(3).unsqueeze(3) # B*21*64*1
        heatmap = 1-dist/opt.ball_radius   # B*21*64*1
        vector = offset/dist.expand(cur_train_size,opt.JOINT_NUM,opt.knn_K,3) # B*21*64*3
        heatmap = torch.cat((heatmap,vector),3)   # B*21*64*4
        heatmap_1 = torch.zeros(cur_train_size,opt.JOINT_NUM,points.size(1)-opt.JOINT_NUM,4).cuda() # B*21*1024*4
        heatmap_1 = heatmap_1.scatter_(2, idx_group_l1_long_long, heatmap) #heatmap_1 = B*21*1024*4
        #heatmap_1[:,idx_group_l1_long_long,idx_group_l1_long_long,:] = heatmap #heatmap_1 = B*21*1024*4
		
        heatmap = heatmap_1.permute(0,1,3,2) #B*21*4*1024
       #print(heatmap.shape)
        heatmap = heatmap.contiguous().view(cur_train_size,opt.JOINT_NUM*4,points.size(1)-opt.JOINT_NUM) #B*4J*1024
		#heatmap_1 = torch.zeros(cur_train_size,opt.JOINT_NUM*4,1024) #B*4J*(1024)
        #append = torch.zeros(cur_train_size,opt.JOINT_NUM*4,1024-opt.knn_K) #B*4J*(1024-64)
        #heatmap = torch.cat((heatmap,append),2) #B*4J*1024
        heatmap = heatmap.unsqueeze(3) #B*4J*1024*1
        return heatmap


def jitter(points, var=0.01, clip=0.05):
    device = points.device
    B, N, C = points.shape

    noise = torch.clamp(torch.normal(0.0, var, [B, N, 3]), -1 * clip, clip).to(device)

    points[:,:, 0:3] = points[:,:, 0:3] + noise
    return points

def rotate_point_cloud_by_random(batch_data, batch_gt, normal=False):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    device = batch_data.device
    B, N, C = batch_data.shape

    for k in range(B):
        # ICVL
        rotation_angle = np.random.rand() * 2 * np.pi
        # MSRA
        # rotation_angle = rotation_region / 48.0 * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0,  1]], dtype=np.float32)

        rotation_matrix = torch.from_numpy(rotation_matrix).to(device)
        batch_data[k, :, 0:3] = torch.mm(batch_data[k, :, 0:3], rotation_matrix)
        batch_gt[k, :, 0:3] = torch.mm(batch_gt[k, :, 0:3], rotation_matrix)
        if normal is True:
            batch_data[k, :, 3:6] = torch.mm(batch_data[k, :, 3:6], rotation_matrix)
    return batch_data, batch_gt


def rotate_point_cloud_by_angle(batch_data, batch_gt, normal=False):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    device = batch_data.device
    B, N, C = batch_data.shape

    for k in range(B):
        rotation_region = np.random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        # ICVL
        rotation_angle = rotation_region / 48.0 * 2 * np.pi
        # MSRA
        # rotation_angle = rotation_region / 48.0 * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, cosval, -sinval],
                                    [0, sinval,  cosval]], dtype=np.float32)

        rotation_matrix = torch.from_numpy(rotation_matrix).to(device)
        batch_data[k, :, 0:3] = torch.mm(batch_data[k, :, 0:3], rotation_matrix)
        batch_gt[k, :, 0:3] = torch.mm(batch_gt[k, :, 0:3], rotation_matrix)
        if normal is True:
            batch_data[k, :, 3:6] = torch.mm(batch_data[k, :, 3:6], rotation_matrix)
    return batch_data, batch_gt

def rotate_point_cloud_by_angle_flip(batch_data, batch_gt, normal=False):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    device = batch_data.device
    B, N, C = batch_data.shape

    for k in range(B):
        rotation_region = np.random.choice([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # ICVL
        rotation_angle = rotation_region / 48.0 * 2 * np.pi
        # MSRA
        # rotation_angle = rotation_region / 48.0 * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0,  1]], dtype=np.float32)
        if np.random.rand() > 0.5:
            rotation_matrix = np.dot(rotation_matrix, np.array([[-1, 0, 0],
                                                                [0, -1, 0],
                                                                [0, 0,  1]], dtype=np.float32))
        rotation_matrix = torch.from_numpy(rotation_matrix).to(device)
        batch_data[k, :, 0:3] = torch.mm(batch_data[k, :, 0:3], rotation_matrix)
        batch_gt[k, :, 0:3] = torch.mm(batch_gt[k, :, 0:3], rotation_matrix)
        if normal is True:
            batch_data[k, :, 3:6] = torch.mm(batch_data[k, :, 3:6], rotation_matrix)
    
    return batch_data, batch_gt

def rotate_point_cloud_by_angle_xyz(batch_data, batch_gt, normal=False):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    device = batch_data.device
    B, N, C = batch_data.shape

    for k in range(B):
        rand_num = np.random.rand()
        # if rand_num < 0.25:
        if rand_num < 0.4:
            rotation_region = np.random.choice(49) - 24
            rotation_angle = rotation_region / 72.0 * 2 * np.pi

            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, -sinval, 0],
                                        [sinval, cosval, 0],
                                        [0, 0,  1]], dtype=np.float32)
            if np.random.rand() > 0.5:
                rotation_matrix = np.dot(rotation_matrix, np.array([[-1, 0, 0],
                                                                    [0, -1, 0],
                                                                    [0, 0,  1]], dtype=np.float32))
        # elif 0.25 < rand_num and rand_num < 0.5:
        elif 0.4 < rand_num and rand_num < 0.7:
            rotation_region = np.random.choice(25) - 12
            rotation_angle = rotation_region / 144.0 * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]], dtype=np.float32)
        # elif 0.5 < rand_num and rand_num < 0.75:
        else:
            rotation_region = np.random.choice(25) - 12
            rotation_angle = rotation_region / 144.0 * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[1, 0, 0],
                                        [0,cosval, -sinval],
                                        [0, sinval, cosval],
                                        ], dtype=np.float32)

        rotation_matrix = torch.from_numpy(rotation_matrix).to(device)
        batch_data[k, :, 0:3] = torch.mm(batch_data[k, :, 0:3], rotation_matrix)
        batch_gt[k, :, 0:3] = torch.mm(batch_gt[k, :, 0:3], rotation_matrix)
        if normal is True:
            batch_data[k, :, 3:6] = torch.mm(batch_data[k, :, 3:6], rotation_matrix)
    
    return batch_data, batch_gt



def rotate_point_cloud_by_angle_nyu(batch_data, batch_gt, normal=False):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    device = batch_data.device
    B, N, C = batch_data.shape

    for k in range(B):
        rand_num = np.random.rand()

        # if rand_num < 0.25:
        if rand_num < 0.4:
            rotation_region = np.random.choice([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            rotation_angle = rotation_region / 48.0 * 2 * np.pi

            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, -sinval, 0],
                                        [sinval, cosval, 0],
                                        [0, 0,  1]], dtype=np.float32)
            if np.random.rand() > 0.5:
                rotation_matrix = np.dot(rotation_matrix, np.array([[-1, 0, 0],
                                                                    [0, -1, 0],
                                                                    [0, 0,  1]], dtype=np.float32))
        # elif 0.25 < rand_num and rand_num < 0.5:
        elif 0.4 < rand_num and rand_num < 0.7:
            rotation_region = np.random.choice([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            rotation_angle = rotation_region / 108.0 * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]], dtype=np.float32)
        # elif 0.5 < rand_num and rand_num < 0.75:
        else:
            rotation_region = np.random.choice([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            rotation_angle = rotation_region / 108.0 * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[1, 0, 0],
                                        [0,cosval, -sinval],
                                        [0, sinval, cosval],
                                        ], dtype=np.float32)
        # # else:
        # rotation_region = np.random.choice([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # rotation_angle = rotation_region / 48.0 * 2 * np.pi

        # cosval = np.cos(rotation_angle)
        # sinval = np.sin(rotation_angle)
        # rotation_matrix = np.array([[cosval, -sinval, 0],
        #                             [sinval, cosval, 0],
        #                             [0, 0,  1]], dtype=np.float32)
        # if np.random.rand() > 0.5:
        #     rotation_matrix = np.dot(rotation_matrix, np.array([[-1, 0, 0],
        #                                                         [0, -1, 0],
        #                                                         [0, 0,  1]], dtype=np.float32))
        # rotation_region = np.random.choice([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # rotation_angle = rotation_region / 108.0 * 2 * np.pi
        # cosval = np.cos(rotation_angle)
        # sinval = np.sin(rotation_angle)
        # rotation_matrix_x = np.array([[1, 0, 0],
        #                             [0,cosval, -sinval],
        #                             [0, sinval, cosval],
        #                             ], dtype=np.float32)
        # rotation_matrix = np.dot(rotation_matrix, rotation_matrix_x)

        rotation_matrix = torch.from_numpy(rotation_matrix).to(device)
        batch_data[k, :, 0:3] = torch.mm(batch_data[k, :, 0:3], rotation_matrix)
        batch_gt[k, :, 0:3] = torch.mm(batch_gt[k, :, 0:3], rotation_matrix)
        if normal is True:
            batch_data[k, :, 3:6] = torch.mm(batch_data[k, :, 3:6], rotation_matrix)
    
    return batch_data, batch_gt




def group_points(points, opt, sampling_func='random'):
    # group points using knn and ball query
    # points: B * 1024 * 6
    cur_train_size = len(points)

    permutation = torch.randperm(points.size(1))
    points = points[:,permutation,:]

    centroid_pts = points[:,0:opt.sample_num_level1,0:3]

    inputs1_diff = points[:,:,0:3].transpose(1,2).unsqueeze(1).expand(cur_train_size,opt.sample_num_level1,3,opt.SAMPLE_NUM) \
                 - centroid_pts[:,:,0:3].unsqueeze(-1).expand(cur_train_size,opt.sample_num_level1,3,opt.SAMPLE_NUM)# B * 512 * 3 * 1024
    inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)    # B * 512 * 3 * 1024
    inputs1_diff = inputs1_diff.sum(2)                      # B * 512 * 1024
    dists, inputs1_idx = torch.topk(inputs1_diff, opt.knn_K, 2, largest=False, sorted=False)  # dists: B * 512 * 64; inputs1_idx: B * 512 * 64
        
    # ball query
    # invalid_map = dists.gt(opt.ball_radius) # B * 512 * 64
    # for jj in range(opt.sample_num_level1):
    #     inputs1_idx[:,jj,:][invalid_map[:,jj,:]] = jj
        
    idx_group_l1_long = inputs1_idx.view(cur_train_size,opt.sample_num_level1*opt.knn_K,1).expand(cur_train_size,opt.sample_num_level1*opt.knn_K,opt.INPUT_FEATURE_NUM)
    inputs_level1 = points.gather(1,idx_group_l1_long).view(cur_train_size,opt.sample_num_level1,opt.knn_K,opt.INPUT_FEATURE_NUM) # B*512*64*6

    inputs_level1_center = centroid_pts[:,:,0:3].unsqueeze(-2)       # B*512*1*3

    inputs_level1[:,:,:,0:3] = inputs_level1[:,:,:,0:3] - inputs_level1_center.expand(cur_train_size,opt.sample_num_level1,opt.knn_K,3)
    inputs_level1 = inputs_level1.unsqueeze(1).transpose(1,4).squeeze(4)  # B*6*512*64
    inputs_level1_center = inputs_level1_center.contiguous().view(-1,1,opt.sample_num_level1,3).transpose(1,3)  # B*3*512*1
    return inputs_level1, inputs_level1_center
    #inputs_level1: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, inputs_level1_center: B*3*sample_num_level1*1
    
def group_points_2(points, sample_num_level1, sample_num_level2, knn_K, ball_radius, sampling_func='random'):
    # group points using knn and ball query
    # points: B*(3+128)*512
    cur_train_size = points.size(0)

    # permutation = torch.randperm(points.size(2))
    permutation = torch.randperm(points.size(2))
    points = points[:,:,permutation]
    centroid_pts = points[:,:,0:sample_num_level2]


    inputs1_diff = points[:,0:3,:].unsqueeze(1).expand(cur_train_size,sample_num_level2,3,sample_num_level1) \
                 - centroid_pts[:,0:3,:].transpose(1,2).unsqueeze(-1).expand(cur_train_size,sample_num_level2,3,sample_num_level1)# B * 128 * 3 * 512
    inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)    # B * 128 * 3 * 512
    inputs1_diff = inputs1_diff.sum(2)                      # B * 128 * 512
    dists, inputs1_idx = torch.topk(inputs1_diff, knn_K, 2, largest=False, sorted=False)  # dists: B * 128 * 64; inputs1_idx: B * 128 * 64
        
    # ball query
    # invalid_map = dists.gt(ball_radius) # B * 128 * 64, invalid_map.float().sum()
    # #pdb.set_trace()
    # for jj in range(sample_num_level2):
    #     inputs1_idx.data[:,jj,:][invalid_map.data[:,jj,:]] = jj

    idx_group_l1_long = inputs1_idx.view(cur_train_size,1,sample_num_level2*knn_K).expand(cur_train_size,points.size(1),sample_num_level2*knn_K)
    inputs_level2 = points.gather(2,idx_group_l1_long).view(cur_train_size,points.size(1),sample_num_level2,knn_K) # B*131*128*64

    inputs_level2_center = centroid_pts[:,0:3,:].unsqueeze(3)       # B*3*128*1
    
    inputs_level2[:,0:3,:,:] = inputs_level2[:,0:3,:,:] - inputs_level2_center.expand(cur_train_size,3,sample_num_level2,knn_K) # B*3*128*64
    return inputs_level2, inputs_level2_center
    # inputs_level2: B*131*sample_num_level2*knn_K, inputs_level2_center: B*3*sample_num_level2*1

def final_group(joint_coarse, original_points, num_neighbors, ball_radius):
    # group points using knn and ball query
    # points: B * 1024 * 6
    cur_train_size = len(original_points)
    original_length = original_points.size(1)
    joint_length = joint_coarse.size(1)
    feature_width = original_points.size(2)

    diff = original_points[:,:,0:3].transpose(1,2).unsqueeze(1).expand(cur_train_size, joint_length, 3, original_length) \
                 - joint_coarse[:,:,0:3].unsqueeze(-1).expand(cur_train_size, joint_length, 3, original_length)# B * n * 3 * 21
    diff = torch.mul(diff, diff)    # B * n * 3 * 21
    diff = diff.sum(2)              # B * n * 1* 21
    # diff = diff.transpose(1,2)                   # B * 21 * n
    dists, idx = torch.topk(diff, num_neighbors, 2, largest=False, sorted=False)  # dists: B * 21 * n; inputs1_idx: B * 21 * n
        
    # ball query
    # invalid_map = dists.gt(ball_radius) # B * 21 * n
    # for jj in range(joint_length):
    #     idx[:,jj,:][invalid_map[:,jj,:]] = jj
        
    idx_group_l1_long = idx.view(cur_train_size,joint_length*num_neighbors, 1).expand(cur_train_size, joint_length*num_neighbors, feature_width)
    grouped_points = original_points.gather(1,idx_group_l1_long).view(cur_train_size, joint_length, num_neighbors, feature_width) # B*21*64*f

    grouped_points[:,:,:,0:3] = grouped_points[:,:,:,0:3] - joint_coarse[:,:,0:3].unsqueeze(-2).expand(cur_train_size, joint_length, num_neighbors, 3)
    
    return grouped_points
    #inputs_level1: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, inputs_level1_center: B*3*sample_num_level1*1

def knn(point_center, points_samples, num_neighbors):
    cur_train_size = len(point_center)
    center_length = point_center.size(1)
    sample_length = points_samples.size(1)
    feature_width = point_center.size(2)

    diff = point_center[:,:,0:3].transpose(1,2).unsqueeze(1).expand(cur_train_size, sample_length, 3, center_length) \
                 - points_samples[:,:,0:3].unsqueeze(-1).expand(cur_train_size, sample_length, 3, center_length)# B * n * 3 * 21
    diff = torch.mul(diff, diff)    # B * n * 3 * 21
    diff = diff.sum(2)              # B * n * 1* 21
    # diff = diff.transpose(1,2)                   # B * 21 * n
    dists, idx = torch.topk(diff, num_neighbors, 2, largest=False, sorted=False)  # dists: B * 21 * n; inputs1_idx: B * 21 * n

    return (idx, dists)


import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename='test.png'):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    kpt_3d = kpt_3d[:, [0, 2, 1]]
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1,0], kpt_3d[i2,0]])
        y = np.array([kpt_3d[i1,1], kpt_3d[i2,1]])
        z = np.array([kpt_3d[i1,2], kpt_3d[i2,2]])

        if kpt_3d_vis[i1,0] > 0 and kpt_3d_vis[i2,0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1,0] > 0:
            ax.scatter(kpt_3d[i1,0], kpt_3d[i1,2], -kpt_3d[i1,1], c=colors[l], marker='o')
        if kpt_3d_vis[i2,0] > 0:
            ax.scatter(kpt_3d[i2,0], kpt_3d[i2,2], -kpt_3d[i2,1], c=colors[l], marker='o')

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()
    # cam_equal_aspect_3d(ax, kpt_3d, transpose=True)
    # plt.show()
    # cv2.waitKey(0)

    plt.savefig(filename)

def cam_equal_aspect_3d(ax, verts, flip_x=False, transpose=True):
    '''
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    :param ax:
    :param verts:
    :param flip_x:
    :return:
    '''
    # print(verts.shape)
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    # min_lim, max_lim = np.min(centers - r), np.max(centers + r)
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
        # ax.set_xlim(max_lim, min_lim)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
        # ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(centers[1] - r, centers[1] + r)
    ax.set_zlim(centers[2] + r, centers[2] - r)
    # ax.set_ylim(min_lim, max_lim)
    # ax.set_zlim(max_lim, min_lim)
    if transpose:
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
    else:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    # ax.view_init(5, -5)
    ax.view_init(5, -85)

def display_keypoints(path, keypoints=None, ax=None, transpose=True):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    COLOR = ['#330000', '#660000', '#b30000', '#ff0000', '#ff4d4d', '#ff9999']
    if transpose:
        keypoints = keypoints[:, [0, 2, 1]]
    for i in range(keypoints.shape[0]):
        ax.scatter(keypoints[i, 0], keypoints[i, 1], keypoints[i, 2], color='red')

    cam_equal_aspect_3d(ax, keypoints, transpose=transpose)

    if path:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)