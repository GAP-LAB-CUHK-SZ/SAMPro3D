import numpy as np
import torch
import os
import plyfile
import cv2

"""
main functions
"""

def load_ply(ply_path):
    ply_data = plyfile.PlyData.read(ply_path)
    data = ply_data['vertex']

    data = np.concatenate([data['x'].reshape(1, -1), data['y'].reshape(1, -1), data['z'].reshape(1, -1), \
                        data['red'].reshape(1, -1), data['green'].reshape(1, -1), data['blue'].reshape(1, -1)], axis=0)

    xyz = data.T[:, :3]
    rgb = data.T[:, 3:]
    
    return xyz, rgb


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        print(f'Folder created at "{folder_path}"')
    else:
        print(f'Folder already exists at "{folder_path}"')


def transform_pt_depth_scannet_torch(points, depth_intrinsic, depth, pose, device):
    """
    :param points: N x 3 format
    :param depth: H x W format
    :param intrinsic: 3x3 format
    :return p: N x 2 format
    """

    vis_thres = 0.1
    depth_shift = 1000.0
    
    fx = depth_intrinsic[0,0]
    fy = depth_intrinsic[1,1]
    cx = depth_intrinsic[0,2]
    cy = depth_intrinsic[1,2]
    bx = depth_intrinsic[0,3]
    by = depth_intrinsic[1,3]
    
    points_world = torch.cat([points, torch.ones((points.shape[0], 1), dtype=torch.float64).to(device)], dim=-1).to(torch.float64)
    world_to_camera = torch.inverse(pose)
    
    p = torch.matmul(world_to_camera, points_world.T)  # [Xb, Yb, Zb, 1]: 4, n
    p[0] = ((p[0] - bx) * fx) / p[2] + cx 
    p[1] = ((p[1] - by) * fy) / p[2] + cy
    
    all_idx = torch.arange(0, len(points)).to(device)  # to save the corresponding point idx later as the prompt ID
    # out-of-image check
    idx = torch.unique(torch.cat([torch.where(p[0]<=0)[0], torch.where(p[1]<=0)[0], \
                                    torch.where(p[0]>=depth.shape[1]-1)[0], \
                                    torch.where(p[1]>=depth.shape[0]-1)[0]], dim=0), dim=0)
    keep_idx = all_idx[torch.isin(all_idx, idx, invert=True)]
    p = p[:, keep_idx]

    if p.shape[1] == 0:
        return p, keep_idx  # no 3D prompt is visible in this frame
        
    # Simply round the final coordinates into pixel value
    pi = torch.round(p).to(torch.int64)
    # Check occlusion
    est_depth = p[2]
    trans_depth = depth[pi[1], pi[0]] / depth_shift
    idx_keep = torch.where(torch.abs(est_depth - trans_depth) <= vis_thres)[0]
    
    p = p.T[idx_keep, :2]
    keep_idx = keep_idx[idx_keep]
    
    return p, keep_idx


def compute_mapping(points, data_path, scene_name, frame_id):  
    """
    :param points: N x 3 format
    :param depth: H x W format
    :param intrinsic: 3x3 format
    :return: mapping, N x 3 format, (H,W,mask)
    """
    vis_thres = 0.1
    depth_shift = 1000.0

    mapping = np.zeros((3, points.shape[0]), dtype=int)
    
    # Load the intrinsic matrix
    depth_intrinsic = np.loadtxt(os.path.join(data_path, 'intrinsics.txt'))
    
    # Load the depth image, and camera pose
    depth = cv2.imread(os.path.join(data_path, scene_name, 'depth', frame_id + '.png'), -1) # read 16bit grayscale 
    pose = np.loadtxt(os.path.join(data_path, scene_name, 'pose', frame_id + '.txt' ))

    fx = depth_intrinsic[0,0]
    fy = depth_intrinsic[1,1]
    cx = depth_intrinsic[0,2]
    cy = depth_intrinsic[1,2]
    bx = depth_intrinsic[0,3]
    by = depth_intrinsic[1,3]
    
    points_world = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1)
    world_to_camera = np.linalg.inv(pose)
    p = np.matmul(world_to_camera, points_world.T)  # [Xb, Yb, Zb, 1]: 4, n
    p[0] = ((p[0] - bx) * fx) / p[2] + cx 
    p[1] = ((p[1] - by) * fy) / p[2] + cy
    
    # out-of-image check
    mask = (p[0] > 0) * (p[1] > 0) \
                    * (p[0] < depth.shape[1]-1) \
                    * (p[1] < depth.shape[0]-1)

    pi = np.round(p).astype(int) # simply round the projected coordinates
    
    # directly keep the pixel whose depth!=0
    depth_mask = depth[pi[1][mask], pi[0][mask]] != 0
    mask[mask == True] = depth_mask
    
    # occlusion check:
    trans_depth = depth[pi[1][mask], pi[0][mask]] / depth_shift
    est_depth = p[2][mask]
    occlusion_mask = np.abs(est_depth - trans_depth) <= vis_thres
    mask[mask == True] = occlusion_mask

    mapping[0][mask] = p[1][mask]
    mapping[1][mask] = p[0][mask]
    mapping[2][mask] = 1

    return mapping.T


def isolate_on_pred(xyz, pt_pred, pt_score):
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=0.04, min_samples=1)  # for sparser point cloud data, eps may need to be larger (e.g., 0.08 for matterport)

    ins_preds = np.unique(pt_pred)
    for ins_id in ins_preds:
        if ins_id == -100:
            continue
        pt_id_ins = np.where(pt_pred == int(ins_id))[0]
        if pt_id_ins.shape[0] <= 0:
            pt_score[:, ins_id] = 0
            continue

        xyz_ins = xyz[pt_id_ins]
        cluster_labels = clustering.fit_predict(xyz_ins)
        # Filter out noise points by excluding the points with a cluster_label of -1
        filtered_labels = cluster_labels[cluster_labels != -1]
        # Count the number of points in each cluster
        unique_labels, label_counts = np.unique(filtered_labels, return_counts=True)
        if label_counts.shape[0] == 0:
            pt_score[pt_id_ins, ins_id] = 0
            continue
        # Find the cluster label with the most points:
        if label_counts.shape[0] > 1:
            most_points_cluster_label = unique_labels[np.argmax(label_counts)]
            remove_points_cluster_pt_id = pt_id_ins[np.where(filtered_labels != most_points_cluster_label)]
            pt_score[remove_points_cluster_pt_id, ins_id] = 0

    return pt_score


def isolate_on_score(xyz, pt_score_mean, pt_score_merge):
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=0.04, min_samples=1)  # for sparser point cloud data, eps may need to be larger (e.g., 0.08 for matterport)
    start = 0.
    stop = 1.
    step = 0.1
    i = start

    # set isolate noisy predictions on score space under different score threshold:
    while i < stop:
        i += step
        valid_thres = i
        ins_score_mean = pt_score_mean.T.copy()
        ins_score = pt_score_merge.T.copy()
        for ins_id in range(ins_score.shape[0]):
            pt_id_ins_mean = np.where(ins_score_mean[ins_id] > valid_thres)[0]  # mean_score (probability) is only for thresholding more easily
            pt_id_ins_abs = np.where(ins_score[ins_id] > 0)[0]
            pt_id_ins = pt_id_ins_abs[np.isin(pt_id_ins_abs, pt_id_ins_mean)]
            if pt_id_ins.shape[0] <= 0:
                continue
            xyz_ins = xyz[pt_id_ins]
            cluster_labels = clustering.fit_predict(xyz_ins)
            # Filter out noise points by excluding the points with a cluster_label of -1
            filtered_labels = cluster_labels[cluster_labels != -1]
            # Count the number of points in each cluster
            unique_labels, label_counts = np.unique(filtered_labels, return_counts=True)
            if label_counts.shape[0] == 0:
                pt_score_merge[pt_id_ins, ins_id] = 0
                continue
            if label_counts.shape[0] > 1:
                # Find the cluster label with the most points:
                most_points_cluster_label = unique_labels[np.argmax(label_counts)]
                remove_points_cluster_pt_id = pt_id_ins[np.where(filtered_labels != most_points_cluster_label)]
                pt_score_merge[remove_points_cluster_pt_id, ins_id] = 0
    
    return pt_score_merge


def merge_common_values(list_of_common_values):
    merged_list = []
    common_values_dict = {}

    for sublist in list_of_common_values:
        common_key = None
        for value in sublist:
            if value in common_values_dict:
                common_key = common_values_dict[value]
                break

        if common_key is None:
            common_key = len(merged_list)
            merged_list.append([])

        for value in sublist:
            if value not in common_values_dict:
                common_values_dict[value] = common_key
                merged_list[common_key].append(value)

    return merged_list


def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def voxelize(coord, voxel_size=0.05, hash_type='fnv', mode=0):
    discrete_coord = np.floor(coord / np.array(voxel_size))
    if hash_type == 'ravel':
        key = ravel_hash_vec(discrete_coord)
    else:
        key = fnv_hash_vec(discrete_coord)

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, count = np.unique(key_sort, return_counts=True)
    if mode == 0:  # train mode
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
        idx_unique = idx_sort[idx_select]
        return idx_unique
    else:  # val mode
        return idx_sort, count