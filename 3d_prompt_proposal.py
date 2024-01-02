"""
Script for the stage of 3D Prompt Proposal in the paper

Author: Mutian Xu (mutianxu@link.cuhk.edu.cn) and Xingyilang Yin
"""

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("default")
import os
import cv2
import argparse
import torch
import numpy as np
import open3d as o3d
import pointops
from utils.main_utils import *
from utils.sam_utils import *
from segment_anything import sam_model_registry, SamPredictor
from tqdm import trange


def create_output_folders(args):
    # Create folder to save SAM outputs:
    create_folder(args.sam_output_path)
    # Create folder to save SAM outputs for a specific scene:
    create_folder(os.path.join(args.sam_output_path, args.scene_name))
    # Create subfolder for saving different output types:
    create_folder(os.path.join(args.sam_output_path, args.scene_name, 'points_npy'))
    create_folder(os.path.join(args.sam_output_path, args.scene_name, 'iou_preds_npy'))
    create_folder(os.path.join(args.sam_output_path, args.scene_name, 'masks_npy'))
    create_folder(os.path.join(args.sam_output_path, args.scene_name, 'corre_3d_ins_npy'))


def prompt_init(xyz, rgb, voxel_size, device):
    # Here we only use voxelization to decide the number of fps-sampled points, \
    # since voxel_size is more controllable. We use fps later for prompt initialization
    idx_sort, num_pt = voxelize(xyz, voxel_size, mode=1)
    print("the number of initial 3D prompts:", len(num_pt))
    xyz = torch.from_numpy(xyz).cuda().contiguous()
    o, n_o = len(xyz), len(num_pt)
    o, n_o = torch.cuda.IntTensor([o]), torch.cuda.IntTensor([n_o])
    idx = pointops.farthest_point_sampling(xyz, o, n_o)
    fps_points = xyz[idx.long(), :]
    fps_points = torch.from_numpy(fps_points.cpu().numpy()).to(device=device)
    rgb = rgb / 256.
    rgb = torch.from_numpy(rgb).cuda().contiguous()
    fps_colors = rgb[idx.long(), :]
    
    return fps_points, fps_colors
    

def save_init_prompt(xyz, rgb, args):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz.cpu().numpy())
    point_cloud.colors = o3d.utility.Vector3dVector(rgb.cpu().numpy())
    prompt_ply_file = os.path.join(args.prompt_path, args.scene_name + '.ply')
    o3d.io.write_point_cloud(prompt_ply_file, point_cloud)
    
    
def process_batch(
    predictor,
    points: torch.Tensor,
    ins_idxs: torch.Tensor,
    im_size: Tuple[int, ...],
) -> MaskData:
    transformed_points = predictor.transform.apply_coords_torch(points, im_size)
    in_points = torch.as_tensor(transformed_points, device=predictor.device)
    in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)

    masks, iou_preds, _ = predictor.predict_torch(
        in_points[:, None, :],
        in_labels[:, None],
        multimask_output=False,
        return_logits=True,
    )
    
    # Serialize predictions and store in MaskData  
    data_original = MaskData(
        masks=masks.flatten(0, 1),
        iou_preds=iou_preds.flatten(0, 1),
        points=points, 
        corre_3d_ins=ins_idxs 
    )

    return data_original
    

def sam_seg(predictor, frame_id_init, frame_id_end, init_prompt, args):
    for i in trange(frame_id_init, frame_id_end):
        frame_id = i
        image = cv2.imread(os.path.join(args.data_path, args.scene_name, 'color', str(frame_id) + '.jpg'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Load the intrinsic
        depth_intrinsic = torch.tensor(np.loadtxt(os.path.join(args.data_path, 'intrinsics.txt')), dtype=torch.float64).to(device=predictor.device)
        # Load the depth, and pose
        depth = cv2.imread(os.path.join(args.data_path, args.scene_name, 'depth', str(frame_id) + '.png'), -1) # read 16bit grayscale 
        depth = torch.from_numpy(depth.astype(np.float64)).to(device=predictor.device)
        pose = torch.tensor(np.loadtxt(os.path.join(args.data_path, args.scene_name, 'pose', str(frame_id) + '.txt')), dtype=torch.float64).to(device=predictor.device)
        
        if str(pose[0, 0].item()) == '-inf': # skip frame with '-inf' pose
            print(f'skip frame {frame_id}')
            continue

        # 3D-2D projection
        input_point_pos, corre_ins_idx = transform_pt_depth_scannet_torch(init_prompt, depth_intrinsic, depth, pose, predictor.device)  # [valid, 2], [valid]
        if input_point_pos.shape[0] == 0 or input_point_pos.shape[1] == 0:
            print(f'skip frame {frame_id}')
            continue

        image_size = image.shape[:2]
        predictor.set_image(image)
        # SAM segmetaion on image
        data_original = MaskData()
        for (points, ins_idxs) in batch_iterator(64, input_point_pos, corre_ins_idx):
            batch_data_original = process_batch(predictor, points, ins_idxs, image_size)
            data_original.cat(batch_data_original)
            del batch_data_original
        predictor.reset_image()
        data_original.to_numpy()

        save_file_name = str(frame_id) + ".npy"
        np.save(os.path.join(args.sam_output_path, args.scene_name, "points_npy", save_file_name), data_original["points"])
        np.save(os.path.join(args.sam_output_path, args.scene_name, "masks_npy", save_file_name), data_original["masks"])  
        np.save(os.path.join(args.sam_output_path, args.scene_name, "iou_preds_npy", save_file_name), data_original["iou_preds"])  
        np.save(os.path.join(args.sam_output_path, args.scene_name, "corre_3d_ins_npy", save_file_name), data_original["corre_3d_ins"])


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate 3d prompt proposal on ScanNet.")
    # for voxelization to decide the number of fps-sampled points:
    parser.add_argument('--voxel_size', default=0.2, type=float, help='Size of voxels.')
    # path arguments:
    parser.add_argument('--data_path', default="dataset/scannet", type=str, help='Path to the dataset containing ScanNet 2d frames and 3d .ply files.')
    parser.add_argument('--scene_name', default="scene0030_00", type=str, help='The scene names in ScanNet.')
    parser.add_argument('--prompt_path', default="init_prompt", type=str, help='Path to the save the sampled 3D initial prompts.')
    parser.add_argument('--sam_output_path', default="sam_output", type=str, help='Path to save the sam segmentation result.')
    # sam arguments:
    parser.add_argument('--model_type', default="vit_h", type=str, help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']")
    parser.add_argument('--sam_checkpoint', default="sam_vit_h_4b8939.pth", type=str, help='The path to the SAM checkpoint to use for mask generation.')
    parser.add_argument("--device", default="cuda:0", type=str, help="The device to run generation on.")
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    args = get_args()
    print("Arguments:")
    print(args)
    # Initialize SAM:
    device = torch.device(args.device)
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint).to(device=device)
    predictor = SamPredictor(sam)
    # Load all 3D points of the input scene:
    scene_plypath = os.path.join(args.data_path, args.scene_name, args.scene_name + '_vh_clean_2.ply')
    xyz, rgb = load_ply(scene_plypath)

    # 3D prompt initialization:
    init_prompt, init_color = prompt_init(xyz, rgb, args.voxel_size, device)
    # save the initial 3D prompts for later use:
    create_folder(args.prompt_path)
    save_init_prompt(init_prompt, init_color, args)

    # SAM segmentation on image frames:
    # create folder to save diffrent SAM output types for later use (note that this is the only stage to perform SAM):
    create_output_folders(args)  # we use npy files to save different output types for faster i/o and clear split
    # perform SAM on each 2D RGB frame:
    frame_id_init = 0
    frame_id_end = len(os.listdir(os.path.join(args.data_path, args.scene_name, 'depth'))) 
    # You can define frame_id_init and frame_id_end by yourself for segmenting partial point clouds from limited frames. Sometimes partial result is better!
    print("Start performing SAM segmentations on {} 2D frames...".format(frame_id_end))
    sam_seg(predictor, frame_id_init, frame_id_end, init_prompt, args)
    print("Finished performing SAM segmentations!")