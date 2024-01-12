"""
Main Script (including 2D-Guided Prompt Filter, Prompt Consolidation, 3D Segmentaiton in the paper)

Author: Mutian Xu (mutianxu@link.cuhk.edu.cn)
"""

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("default")
import os
import sys
import numpy as np
import torch
import argparse
from natsort import natsorted 
import open3d as o3d
from tqdm import tqdm

from utils.sam_utils import *
from utils.main_utils import *
from utils.vis_utils import *
from segment_anything import sam_model_registry, SamPredictor


# 2D-Guided Prompt Filter:
def prompt_filter(init_prompt, scene_output_path, npy_path, predictor, args):
    device = torch.device(args.device)
    # gap = 1  # number of skipped frames
    stop_limit = 10  # we find that not considering all frames for filter is better

    keep_score = torch.zeros(len(init_prompt), device=device)
    counter = torch.zeros(len(init_prompt), device=device)
    del_score = torch.zeros(len(init_prompt), device=device)

    for i, (npy_file) in enumerate(tqdm(npy_path)):
        # if i != 0 and i % gap != 0:
        #     continue

        # load the corresponding SAM segmentations data of the corresponding frame:
        points_data = torch.from_numpy(np.load(os.path.join(scene_output_path, 'points_npy', npy_file))).to(device)
        iou_preds_data = torch.from_numpy(np.load(os.path.join(scene_output_path, 'iou_preds_npy', npy_file))).to(device)
        masks_data = torch.from_numpy(np.load(os.path.join(scene_output_path, 'masks_npy', npy_file))).to(device)
        corre_3d_ins_data = torch.from_numpy(np.load(os.path.join(scene_output_path, 'corre_3d_ins_npy', npy_file))).to(device)  # the valid (i.e., has mapped pixels at the current frame) prompt ID  in the original 3D point cloud of initial prompts
        data = MaskData(
                masks=masks_data,
                iou_preds=iou_preds_data,
                points=points_data, 
                corre_3d_ins=corre_3d_ins_data 
            )

        corr_ins_idx = data['corre_3d_ins']
        # ins_flag[corr_ins_idx] = 1 # set the valid ins value to 1
        counter[corr_ins_idx] += 1  # only count if it is not the stopped instances
        stop_id = torch.where(counter >= stop_limit)[0]

        ############ start filter:
        # Filter by predicted IoU
        if args.pred_iou_thres > 0.0:
            keep_mask = data["iou_preds"] > args.pred_iou_thres
            data.filter(keep_mask)
        #     print(data['points'].shape)
        
        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            masks=data["masks"], mask_threshold=predictor.model.mask_threshold, threshold_offset=1.0
        )

        if args.stability_score_thres > 0.0:
            keep_mask = data["stability_score"] >= args.stability_score_thres
            data.filter(keep_mask)
    #     print(data['points'].shape)
        
        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        
        # Remove duplicates within this crop.
        from torchvision.ops.boxes import batched_nms, box_area 
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=args.box_nms_thres,
        )
        data.filter(keep_by_nms)

        keep_ins_idx = data["corre_3d_ins"]
        del_ins_idx = corr_ins_idx[torch.isin(corr_ins_idx, keep_ins_idx, invert=True)]

        if stop_id.shape[0] > 0:
            keep_ins_idx = keep_ins_idx[torch.isin(keep_ins_idx, stop_id, invert=True)]
            del_ins_idx = del_ins_idx[torch.isin(del_ins_idx, stop_id, invert=True)]
        keep_score[keep_ins_idx] += 1
        del_score[del_ins_idx] += 1 

    # make all selected frames happy:
    counter[torch.where(counter >= stop_limit)] = stop_limit
    counter[torch.where(counter == 0)] = -1  #  avoid that the the score is divided by counter of 0
    # keep prompts whose score is larger than a threshold:
    keep_score_mean = keep_score / counter
    keep_idx = torch.where(keep_score_mean >= args.keep_thres)[0]

    print("the number of prompts after filter", keep_idx.shape[0])

    return keep_idx


def perform_3dsegmentation(xyz, keep_idx, scene_output_path, npy_path, args):
    device = torch.device(args.device)
    # gap = 1  # number of skipped frames
    n_points = xyz.shape[0]
    num_ins = keep_idx.shape[0]
    pt_score = torch.zeros([n_points, num_ins], device=device)  # All input points have a score
    counter_final = torch.zeros([n_points, num_ins], device=device)

    for i, (npy_file) in enumerate(tqdm(npy_path)):
        # if i != 0 and i % gap != 0:
        #     continue

        # load the corresponding SAM segmentations data of the corresponding frame:
        points_data = torch.from_numpy(np.load(os.path.join(scene_output_path, 'points_npy', npy_file))).to(device)
        iou_preds_data = torch.from_numpy(np.load(os.path.join(scene_output_path, 'iou_preds_npy', npy_file))).to(device)
        masks_data = torch.from_numpy(np.load(os.path.join(scene_output_path, 'masks_npy', npy_file))).to(device)
        corre_3d_ins_data = torch.from_numpy(np.load(os.path.join(scene_output_path, 'corre_3d_ins_npy', npy_file))).to(device)  # the valid (i.e., has mapped pixels at the current frame) prompt ID  in the original 3D point cloud of initial prompts
        data = MaskData(
                masks=masks_data,
                iou_preds=iou_preds_data,
                points=points_data, 
                corre_3d_ins=corre_3d_ins_data)

        frame_id = npy_file[:-4]

        # calculate the 3d-2d mapping on ALL input points (not just prompt)
        mapping = compute_mapping(xyz, args.data_path, args.scene_name, frame_id)
        if mapping[:, 2].sum() == 0: # no points corresponds to this image, skip
            continue
        mapping = torch.from_numpy(mapping).to(device)

        keep_mask = torch.isin(data["corre_3d_ins"], keep_idx)  # only keep the mask that has been kept during previous prompt filter process
        data.filter(keep_mask)

        masks_logits = data["masks"]
        masks = masks_logits > 0.

        ins_idx_all = []
        for actual_idx in data["corre_3d_ins"]:  # the actual prompt ID in the original 3D point cloud of initial prompts, \
            # for calculating pt_score later (since pt_score is considered on all initial prompts)
            ins_idx = torch.where(keep_idx == actual_idx)[0]
            ins_idx_all.append(ins_idx.item())
        
        # when both a point i and a prompt j is found in this frame, counter[i, j] + 1
        counter_point = mapping[:, 2]   # the found points
        counter_point = counter_point.reshape(-1, 1).repeat(1, num_ins)
        counter_ins = torch.zeros(num_ins, device=device)
        counter_ins[ins_idx_all] += 1   # the found prompts
        counter_ins = counter_ins.reshape(1, -1).repeat(n_points, 1)
        counter_final += (counter_point * counter_ins)

        # caculate the score on mask area:
        for index, (mask) in enumerate(masks):  # iterate over each mask area segmented by different prompts
            ins_id = ins_idx_all[index]  # get the actual instance id  # ins_idx_al
            mask = mask.int()
        
            mask_2d_3d = mask[mapping[:, 0], mapping[:, 1]]
            mask_2d_3d = mask_2d_3d * mapping[:, 2]  # set the score to 0 if no mapping is found
            
            pt_score[:, ins_id] += mask_2d_3d  # For each individual input point in the scene, \
            # if it is projected within the mask area segmented by a prompt k at current frame, we assign its prediction as the prompt ID k

    pt_score_cpu = pt_score.cpu().numpy()
    counter_final_cpu = counter_final.cpu().numpy()
    counter_final_cpu[np.where(counter_final_cpu==0)] = -1  # avoid divided by zero

    pt_score_mean = pt_score_cpu / counter_final_cpu  # mean score denotes the probability of a point assigned to a specified prompt ID, and is only used for later thresholding
    pt_score_abs = pt_score_cpu
    max_score = np.max(pt_score_mean, axis=-1)  # the actual scores that has been segmented into one instance
    max_score_abs = np.max(pt_score_abs, axis=-1)

    # if pt_score_mean has the max value on more than one instance，we use the instance with higher pt_score as the pred
    max_indices_mean = np.where(pt_score_mean == max_score[:, np.newaxis])
    pt_score_mean_new = pt_score_mean.copy()   # only for calculate label, merge will still use pt_score_mean
    pt_score_mean_new[max_indices_mean] += pt_score_cpu[max_indices_mean]
    pt_pred_mean = np.argmax(pt_score_mean_new, axis=-1) # the ins index

    pt_pred_abs = np.argmax(pt_score_abs, axis=-1)

    low_pt_idx_mean = np.where(max_score <= 0.)[0]  # assign ins_label=-1 (unlabelled) if its score=0 (i.e., no 2D mask assigned)
    pt_score_mean[low_pt_idx_mean] = 0.
    pt_pred_mean[low_pt_idx_mean] = -1

    low_pt_idx_abs = np.where(max_score_abs <= 0.)[0]
    pt_score_abs[low_pt_idx_abs] = 0.
    pt_pred_abs[low_pt_idx_abs] = -1

    return pt_score_abs, pt_pred_abs, pt_score_mean


def prompt_consolidation(xyz, pt_score_abs, pt_pred_abs, pt_score_mean):
    pt_pred_final = pt_pred_abs.copy()

    # for each segmentated space, we first use DBSCAN to separate noisy predictions that are isolated in 3D space. (This aims to refine the SAM results)
    pt_score_merge = isolate_on_pred(xyz, pt_pred_abs.copy(), pt_score_abs.copy())
    pt_score_mean_ori = pt_score_mean.copy()
    pt_score_merge_ori = pt_score_merge.copy()

    # for each segmentated space, we again use DBSCAN to separate noisy score-level predictions (indicating a point has been segmented to a label at one frame) \
    # that are isolated in 3D space. (This aims to refine the SAM results)
    pt_score_merge = isolate_on_score(xyz, pt_score_mean_ori, pt_score_merge_ori)

    # only regard "confident" (label probability > 0.5) points as valid points belonging to an instance (or prompt) for consolidation:
    valid_thres = 0.5
    ins_areas = []
    ins_ids = []
    ins_score_mean = pt_score_mean.T
    ins_score = pt_score_merge.T
    for ins_id in range(ins_score.shape[0]):
        ins_area_mean = np.where(ins_score_mean[ins_id] >= valid_thres)[0]  # mean_score (probability) is only for thresholding more easily
        ins_area_abs = np.where(ins_score[ins_id] > 0)[0]
        ins_area = ins_area_abs[np.isin(ins_area_abs, ins_area_mean)]
        if ins_area.shape[0] > 0:
            ins_areas.append(ins_area)  # append the valid point idx of each instance/prompt
            ins_ids.append(ins_id)

    inter_all = []  # the intersection list to denote which prompts are segmenting the same 3D object
    for i in range(len(ins_areas)):
        inter_ins = [ins_ids[i]]
        for j in range(i+1, len(ins_areas)):
            inter = np.intersect1d(ins_areas[i], ins_areas[j])
            inter_ratio = inter.shape[0] / ins_areas[i].shape[0]
            if inter_ratio > 0.1:  # consider i and j are segmenting the same 3D object if have a certain overlap \
                # and append together in a sublist which are started from i:
                inter_ins.append(ins_ids[j])
            inter_all.append(inter_ins)

    consolidated_list = merge_common_values(inter_all)  # consolidate all prompts (i, j, k, ...) that are segmenting the same 3D object
    print("number of instances after Prompt Consolidation", len(consolidated_list))
        
    # Consolidate the result:
    for sublist in consolidated_list:
        for consolidate_id in sublist:
            mask = np.isin(pt_pred_final, sublist)
            pt_pred_final[mask] = sublist[0]  # regard the first prompt id as the pseudo prompt id

    return pt_pred_final


def merge_floor(pred_ins, floor_propose_ids, floor_id, scene_inter_thres):
    unique_pre_ins_ids = np.unique(pred_ins)
    for i in range(len(unique_pre_ins_ids)):
        if unique_pre_ins_ids[i] == -1:
            pre_instance_points_idx = np.where(pred_ins == unique_pre_ins_ids[i])[0]
            insection = np.isin(pre_instance_points_idx, floor_propose_ids) # the intersection between the floor and the predicted instance
            if sum(insection) > 0: 
                pred_ins[pre_instance_points_idx[insection]] = floor_id
            continue
        
        pre_instance_points_idx = np.where(pred_ins == unique_pre_ins_ids[i])[0]
        insection = sum(np.isin(pre_instance_points_idx, floor_propose_ids))  # the intersection between the floor and the predicted instance
     
        ratio = insection / len(pre_instance_points_idx)
        if ratio > scene_inter_thres:
            pred_ins[pre_instance_points_idx] = floor_id
            print(unique_pre_ins_ids[i])

    return pred_ins


def ransac_plane_seg(scene_plypath, pred_ins, floor_id, scene_dist_thres):
    point_cloud = o3d.io.read_point_cloud(scene_plypath)
    plane, inliers = point_cloud.segment_plane(distance_threshold=scene_dist_thres, ransac_n=3, num_iterations=1000)
    pred_ins[inliers] = floor_id

    return pred_ins


def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser()
    # path arguments:
    parser.add_argument('--data_path', default="dataset/scannet", type=str, help='Path to the dataset containing ScanNet 2d frames and 3d .ply files.')
    parser.add_argument('--scene_name', default="scene0030_00", type=str, help='The scene names in ScanNet.')
    parser.add_argument('--prompt_path', default="init_prompt", type=str, help='Path to the pre-sampled 3D initial prompts.')
    parser.add_argument('--sam_output_path', default="sam_output", type=str, help='Path to the previously generated sam segmentation result.')
    parser.add_argument('--pred_path', default="final_pred", type=str, help='Path to save the predicted per-point segmentation.')
    parser.add_argument('--output_vis_path', default="output_vis", type=str, help='Path to save the visualization file of the final segmentation result.')
    # sam arguments:
    parser.add_argument('--model_type', default="vit_h", type=str, help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']")
    parser.add_argument('--sam_checkpoint', default="sam_vit_h_4b8939.pth", type=str, help='The path to the SAM checkpoint to use for mask generation.')
    parser.add_argument("--device", default="cuda:0", type=str, help="The device to run generation on.")
    # arguments for prompt filter:
    parser.add_argument('--pred_iou_thres', type=float, default=0.7, help='Exclude masks with a predicted score from the model that is lower than this threshold.')
    parser.add_argument('--stability_score_thres', type=float, default=0.6, help='Exclude masks with a stability score lower than this threshold.')
    parser.add_argument('--box_nms_thres', type=float, default=0.8, help='The overlap threshold for excluding a duplicate mask.')
    parser.add_argument('--keep_thres', type=float, default=0.4, help='The keeping threshold for keeping a prompt.')
    # arguments for post-processing floor:
    parser.add_argument('--post_floor', type=bool, default=True, help='Whether post-processing the floor')
    parser.add_argument('--scene_ht_thres', type=float, default=0.08, help='Height threshold of the floor area proposal')
    parser.add_argument('--scene_inter_thres', type=float, default=0.4, help='Intersection threshold')
    parser.add_argument('--scene_dist_thres', type=float, default=0.01, help='Distance_threshold, like a "thickness" of the floor')   
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print("Arguments:")
    print(args)

    # Initialize SAM:
    sys.path.append("..")
    device = torch.device(args.device)
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    print("Start loading SAM segmentations and other meta data ...")
    # Load the initial 3D input prompts (i.e., fps-sampled input points):
    prompt_ply_file = os.path.join(args.prompt_path, args.scene_name + '.ply')
    init_prompt, _ = load_ply(prompt_ply_file)
    print("the number of initial prompts", init_prompt.shape[0])

    # Load all 3D points of the input scene:
    scene_plypath = os.path.join(args.data_path, args.scene_name, args.scene_name + '_vh_clean_2.ply')
    xyz, rgb = load_ply(scene_plypath)

    # Load SAM segmentations generated by previous 3D Prompt Proposal: 
    scene_output_path = os.path.join(args.sam_output_path, args.scene_name)
    points_npy_path = natsorted(os.listdir(os.path.join(scene_output_path, 'points_npy')))
    iou_preds_npy_path = natsorted(os.listdir(os.path.join(scene_output_path, 'iou_preds_npy')))
    masks_npy_path = natsorted(os.listdir(os.path.join(scene_output_path, 'masks_npy')))
    corre_3d_ins_npy_path = natsorted(os.listdir(os.path.join(scene_output_path, 'corre_3d_ins_npy')))
    assert(points_npy_path == iou_preds_npy_path == masks_npy_path == corre_3d_ins_npy_path)
    print("Finished loading SAM segmentations and other meta data!")
    print("********************************************************")

    # 2D-Guided Prompt Filter:
    print("Start 2D-Guided Prompt Filter ...")
    keep_idx = prompt_filter(init_prompt, scene_output_path, points_npy_path, predictor, args)
    # pt_filtered = pt_init[keep_idx.clone().cpu().numpy()]
    print("Finished 2D-Guided Prompt Filter!")
    print("********************************************************")

    # Now we need to perform 3D segmentation to get the initial segmentation label and per-point segmentation score, aimming to check if they are segmenting the same 3D object:
    print("Start initial 3D segmentation ...")
    pt_score_abs, pt_pred_abs, pt_score_mean = perform_3dsegmentation(xyz, keep_idx, scene_output_path, points_npy_path, args)
    print("Finished initial 3D segmentation!")
    print("********************************************************")

    # Prompt Consolidation:
    print("Start Prompt Consolidation and finalizing 3D Segmentation ...")
    pt_pred = prompt_consolidation(xyz, pt_score_abs, pt_pred_abs, pt_score_mean)
    print("Finished running the whole SAMPro3D!")
    print("********************************************************")

    pt_pred = num_to_natural(pt_pred)
    create_folder(args.pred_path)
    # save the prediction result:
    pred_file = os.path.join(args.pred_path, args.scene_name + '_seg.npy')
    np.save(pred_file, pt_pred)

    # Post process for perfect floor segmentation:
    if args.post_floor:
        print("Start post-processing the floor ...")
        # Generate floor instance proposal (min height of the current scene + scene_height_threshold)
        floor_proposal_masks = xyz[:, 2] < min(xyz[:, 2]) + args.scene_ht_thres  # define an initial area of the floor according to the height
        xyz_id = np.arange(len(xyz))
        floor_proposal_ids = xyz_id[floor_proposal_masks]
        floor_id = int(pt_pred.max()) + 1
        # Merge instances that have large overlap with the floor_proposal
        pt_pred = merge_floor(pt_pred, floor_proposal_ids, floor_id, args.scene_inter_thres)
        # Run RANSAC to finally refine the previous plane segmentation if there are still some actual floor areas does not segmented as floor (this can usually be skipped)
        pt_pred = ransac_plane_seg(scene_plypath, pt_pred, floor_id, args.scene_dist_thres)
        print("Finished post-processing the floor!")
        print("********************************************************")
        # save the prediction result:
        pred_file = os.path.join(args.pred_path, args.scene_name + '_seg_floor.npy')
        np.save(pred_file, pt_pred)
    
    print("Creating the visualization result ...")
    create_folder(args.output_vis_path)
    mesh_ori = o3d.io.read_triangle_mesh(scene_plypath)
    pred_ins_num = int(pt_pred.max())+1
    cmap = rand_cmap(pred_ins_num, type='bright', first_color_black=False, last_color_black=False, verbose=False)
    c_all = get_color_rgb(xyz, pt_pred, rgb, cmap)
    c = np.concatenate(c_all)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = mesh_ori.vertices
    mesh.triangles = mesh_ori.triangles
    mesh.vertex_colors = o3d.utility.Vector3dVector(c)
    output_vis_file = os.path.join(args.output_vis_path, args.scene_name + '_seg.ply')
    o3d.io.write_triangle_mesh(output_vis_file, mesh)
    print("Successfully save the visualization result of final segmentation!")