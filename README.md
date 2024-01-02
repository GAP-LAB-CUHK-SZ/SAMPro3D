<!-- PROJECT LOGO -->

<p align="center">
  <img src="https://mutianxu.github.io/sampro3d/static/images/icon_final.jpg" alt="" width="150" height="50"/>
  <h1 align="center">SAMPro3D: Locating SAM Prompts in 3D 
  
  for Zero-Shot Scene Segmentation</h1>
  <p align="center">
    <a href="https://mutianxu.github.io"><strong>Mutian Xu</strong></a>
    ·
    <strong>Xingyilang Yin</strong></a>
    ·
    <a href="https://lingtengqiu.github.io/"><strong>Lingteng Qiu</strong></a>
    ·
    <a href="https://xueyuhanlang.github.io/"><strong>Yang Liu</strong></a>
    ·
    <a href="https://www.microsoft.com/en-us/research/people/xtong/"><strong>Xin Tong</strong></a>
    ·
    <a href="https://gaplab.cuhk.edu.cn/"><strong>Xiaoguang Han</strong></a>
    <br>
    SSE, CUHKSZ
    ·
    FNii, CUHKSZ
    ·
    Microsoft Research Asia

  </p>
  <!-- <h2 align="center">CVPR 2023</h2> -->
  <h3 align="center"><a href="https://arxiv.org/abs/2311.17707">Paper</a> | <a href="https://mutianxu.github.io/sampro3d/">Project Page</a></h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="https://mutianxu.github.io/sampro3d/static/images/teaser.jpg" alt="Logo" width="80%">
  </a>
</p>

*SAMPro3D* can segment **ANY** 3D indoor scenes <b>WITHOUT</b> training. It achieves higher quality and more diverse segmentation than previous zero-shot or fully supervised approaches, and in many cases even surpasses human-level annotations.
<br>

If you find our code or work helpful, please cite:
```bibtex
@article{xu2023sampro3d,
        title={SAMPro3D: Locating SAM Prompts in 3D for Zero-Shot Scene Segmentation}, 
        author={Mutian Xu and Xingyilang Yin and Lingteng Qiu and Yang Liu and Xin Tong and Xiaoguang Han},
        year={2023},
        journal = {arXiv preprint arXiv:2311.17707}
  }
```

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#-news">News</a>
    </li>
    <li>
      <a href="#requirements-and-installation">Requirements and Installation</a>
    </li>
    <li>
      <a href="#data-preparation">Data Preparation</a>
    </li>
    <li>
      <a href="#run-sampro3d">Run SAMPro3D</a>
    </li>
    <li>
      <a href="#animated-qualitative-comparison">Animated Qualitative Comparison</a>
    </li>
    <li>
      <a href="#segment-your-own-3d-scene">Segment Your Own 3D Scene</a>
    </li>
    <li>
      <a href="#todo">TODO</a>
    </li>
    <li>
      <a href="#contact">Contact</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
  </ol>
</details>

## :loudspeaker: News
- Release code. :fire::fire::fire: (12.31, 2023 UTC)
- The first major revision of code, please use the latest one! :fire::fire::fire: (12.31, 2023 UTC)


## Requirements and Installation

### Hardware requirements
At least 1 GPU to hold around 8000MB. Moreover, it is highly recommended to utilize both a CPU with ample processing power and a disk with fast I/O capabilities. Additionally, the disk needs to be large enough (about 50 MB for a 2D frame of resolution 240*320, totally around 160 GB for 2500 frames of a large-scale scene).

### Software installation
Follow the [installation instruction](installation.md) to install all required packages.

## Data Preparation

Follow the [data pre-processing instruction](dataset_preprocess/README.md) to download and preprocess data.

## Run SAMPro3D

### 3D Prompt Proposal
The initial stage of SAMPro3D involves generating a 3D prompt and executing SAM segmentation, followed by saving the SAM outputs for subsequent stages. To initiate this process, simply run:
```
python 3d_prompt_proposal.py --data_path /PATH_TO/ScanNet_data --scene_name sceneXXXX_XX --prompt_path /PATH_TO/initial_prompt --sam_output_path /PATH_TO/SAM_outputs --device cuda:0
```
This stage will be the only step to perform SAM inference, accounting for the majority of computational time and memory usage within our entire pipeline.

**Note**: This stage will save SAM outputs into `.npy` files for later use. Due to different hardware conditions (CPU and disk), the I/O speed of SAM output files may vary a lot and impact the running time of our pipeline. Please refer to the hardware recommendations mentioned before to prepare your hardware for the best efficiency.

(Optional: *Partial-Area* Segmentation): At this stage, you can also perform 3D segmentation on partial point clouds captured by limited 2D frames, by simply changing the `frame_id_init` and `frame_id_end` at [here](https://github.com/GAP-LAB-CUHK-SZ/SAMPro3D/blob/main/3d_prompt_proposal.py#L169), then running the script. Sometimes this works better than segmenting the whole point clouds (thanks to less complicated scenes and better frame-consistency).

### Finish Segmentation
Next, we will proceed with filtering and consolidating the initial prompts, leveraging the saved SAM outputs generated during the 3D Prompt Proposal phase, to obtain the final 3D segmentations. This can be realized by executing the following command:
```
python main.py --data_path /PATH_TO/ScanNet_data --scene_name sceneXXXX_XX --prompt_path /PATH_TO/initial_prompt --sam_output_path /PATH_TO/SAM_outputs --output_vis_path /PATH_TO/result_visualization --device cuda:0
```
After finishing this, the visualization result of the final 3D segmentation will be saved as `.ply` file in the path specified by `--output_vis_path`.

<!-- ### Post-processing to segment the floor perfectly.
 Using our framework, you can usually get a decent segmentation of the floor. However, for a large-scale floor, you need to run post_process.py for perfect segmentation of floors. -->

## Animated Qualitative Comparison

https://github.com/GAP-LAB-CUHK-SZ/SAMPro3D/assets/48080726/3a459c80-ac17-4750-a763-d477d33640bd

https://github.com/GAP-LAB-CUHK-SZ/SAMPro3D/assets/48080726/ac7aaf2c-9223-4d0e-94c4-cf38413eba74

## :star: Segment Your Own 3D Scene: 
With our advanced framework, you can generate high-quality segmentations on your own 3D scene without the need for training! Here are the steps you can follow:

- Data preparation: Follow the tips mentioned in [data pre-processing instruction](dataset_preprocess/README.md) to prepare your data.
- Familiarize yourself with the instructions: Read and understand the guidelines, and pay attention to any specific recommendations.
- Run SAMPro3D: Execute the segmentation framework on your prepared data. You may also need to adjust some parameters such as *eps* at [here](https://github.com/GAP-LAB-CUHK-SZ/SAMPro3D/blob/main/utils/main_utils.py#L139) according to the code comments. 
- Monitor the segmentation process: Keep an eye on the segmentation process to ensure it is running smoothly. Depending on the size and complexity of your scene, the segmentation may take some time to complete.
- Evaluate the segmentation output: Once the segmentation process is finished, assess the quality of the segmentation results. Check if the segments align with the desired objects or regions in your 3D scene. You may also compare the output to ground truth data or visually inspect the results for accuracy.
- Refine if necessary: If the segmentation output requires improvement or refinement, consider adjusting the parameters or settings of SAMPro3D or applying post-processing techniques to enhance the segmentation quality.
- Analyze and utilize the segmentation: Utilize the segmented output for your intended purposes, such as further analysis, visualization, or integration with other applications or systems.

## :triangular_flag_on_post: TODO
- [ ]  Add the visualization code for showing the result of SAM3D, Mask3D and ScanNet200's annotations.
- [ ]  Support the jupyter notebook for step-by-step running.
- [ ]  Add the code for incorporating [HQ-SAM](https://github.com/SysCV/sam-hq) and [Mobile-SAM](https://github.com/ChaoningZhang/MobileSAM) in our pipeline.
- [ ]  Support in-website qualitative visualization.
- [ ]  Support more datasets.

## Contact
You are welcome to submit issues, send pull requests, or share some ideas with us. If you have any other questions, please contact Mutian Xu (mutianxu@link.cuhk.edu.cn).

## Acknowledgement
Our code base is partially borrowed or adapted from [SAM](https://github.com/facebookresearch/segment-anything), [OpenScene](https://github.com/pengsongyou/openscene) and [Pointcept](https://github.com/Pointcept).
