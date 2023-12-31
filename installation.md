# Installation Instruction

Start by cloning the repo:
```bash
git clone https://github.com/GAP-LAB-CUHK-SZ/SAMPro3D.git
cd SAMPro3D
```

First of all, you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `sampro3d` as below. For linux, you need to install `libopenexr-dev` before creating the environment.

```bash
sudo apt-get install libopenexr-dev # for linux
conda create -n sampro3d python=3.11 (recommended python version >= 3.10)
conda activate sampro3d
```

Step 1: install PyTorch (we tested on 1.13.0, but the other recent versions should also work):

```bash
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

Step 2: build pointops:

```bash
cd libs/pointops
# usual
python setup.py install
# docker & multi GPU arch
TORCH_CUDA_ARCH_LIST="ARCH LIST" python setup.py install
# e.g. 7.5: RTX 3000; 8.0: a100 More available in: https://developer.nvidia.com/cuda-gpus
TORCH_CUDA_ARCH_LIST="7.5 8.0" python setup.py install
cd ..
```

Step 3: install the [Segment Anything Model](https://github.com/facebookresearch/segment-anything):
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```
Then download the **`default` [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)** and put it into the folder of SAMPro3D. 

(Optional: you may alternatively use the smaller model [ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) or [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) by simply changing the `args.model_type` and `args.sam_checkpoint` when running our program.)

Step 4: install all the remaining dependencies:
```bash
pip install -r requirements.txt
```