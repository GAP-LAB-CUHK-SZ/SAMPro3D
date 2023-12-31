## Data Preparation

### ScanNet
- Download the [ScanNet](http://www.scan-net.org/) v2 dataset, where `.sens` and `_vh_clean_2.ply.` are two major file types that are needed to run our framework.
- For 3D data, just directly use `.ply` file of downloaded ScanNet v2 raw dataset. (*e.g.: scene0030_00_vh_clean_2.ply*)
- Preprocessing 2D RGB-D data by running:
```
python prepare_2d_scannet.py --scannet_path /PATH_TO/scannet/scans --output_path ../../dataset/scannet --frame_skip 1
```

### Folder structure
After running the pre-processing code above, you should organize a data structure like below, to make sure no path error occurs when running later programs.

```
dataset/
│
├── scannet
│   │
│   ├── scene0011_00
│   │   ├── color
│   │   ├── depth
│   │   ├── pose
│   │   └── scene0011_00_vh_clean_2.ply
│   │
│   ├── scene0011_01
│   │   ├── color
│   │   ├── depth
│   │   ├── pose
│   │   └── scene0011_01_vh_clean_2.ply
│   │
│   └── ...
|   |
|   └── intrinsics.txt (fixed intrinsic parameters for all scenes)
```

### Prepare your own 3D scene for segmentation:
Our framework can segment any 3D scene without training. To produce high-quality segmentation of your own 3D scene, you may just need to prepare your own data as:
- Capture or create a 3D scene: Use a 3D modeling software or capture a real-world scene using 3D scanning techniques, photogrammetry, or depth sensors. You need to get the RGBD and 3D (point cloud or mesh) data at this step. **Note**: scanning *dense* view (continuous 2D frame) can produce better 3D segmentation result.
- Clean your data: This may involve removing noise, aligning multiple scans, *etc*. **Note**: high-resolution RGB data make SAM perform well, ultimately leading to better 3D segmentation result.
- Converting the data into a suitable format: Organize the cleaned data as the above folder structure. Additionally, you may also need to adjust your own RGBD files and 3D data according to ScanNet data.