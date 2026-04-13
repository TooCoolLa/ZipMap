## Training Dataset Preparation for ZipMap 

| Dataset | Static / Dynamic | Comments |
|---|---|---|
| [Aria Synthetic Environments](https://www.projectaria.com/datasets/ase) | Static | Class C; only used first 5% of the data |
| [ARKitScenes](https://github.com/apple/ARKitScenes) | Static | Class B |
| [BlendedMVS](https://github.com/YoYo000/BlendedMVS) | Static | Class B |
| [Co3Dv2](https://github.com/facebookresearch/co3d) | Static | Class A |
| [DL3DV](https://github.com/DL3DV-10K/Dataset) | Static | Class B |
| [GTA-SfM](https://github.com/HKUST-Aerial-Robotics/Flow-Motion-Depth) | Static | Class C |
| [Hypersim](https://github.com/apple/ml-hypersim) | Static | Class B |
| [MapFree](https://research.nianticlabs.com/mapfree-reloc-benchmark/dataset) | Static | Class B |
| [MatrixCity](https://city-super.github.io/matrixcity/) | Static | Class C |
| [Matterport3D](https://niessner.github.io/Matterport/) | Static | Class B |
| [MegaDepth](https://www.cs.cornell.edu/projects/megadepth/) | Static | Class B |
| [MidAir](https://midair.ulg.ac.be/) | Static | Class C |
| [MVS-Synth](https://phuang17.github.io/DeepMVS/mvs-synth.html) | Static | Class B |
| [OmniObject3D](https://omniobject3d.github.io/) | Static | Class B |
| [ScanNet](http://www.scan-net.org/ScanNet/) | Static | Class B |
| [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) | Static | Class B |
| [SceneNetRGBD](https://robotvault.bitbucket.io/scenenet-rgbd.html) | Static | Class C |
| [TartanAir](https://theairlab.org/tartanair-dataset/) | Static | Class B |
| [TartanGround](https://tartanair.org/tartanground/) (tartanair_v2) | Static | Class C |
| [Unreal4K](https://github.com/fabiotosi92/SMD-Nets) | Static | Class B |
| [Virtual KITTI](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) | Static | Class A |
| [Waymo](https://github.com/waymo-research/waymo-open-dataset) | Static | Class B |
| [WildRGBD](https://github.com/wildrgbd/wildrgbd/) | Static | Class B |
| [BEDLAM](https://bedlam.is.tue.mpg.de/) | Dynamic | Class B |
| [Dynamic Replica](https://github.com/facebookresearch/dynamic_stereo) | Dynamic | Class B |
| [Kubric](https://github.com/google-research/kubric) | Dynamic | Class C |
| [OmniWorld](https://yangzhou24.github.io/OmniWorld/) | Dynamic | Class C; only used the first released 5k video clips of the data |
| [PointOdyssey](https://pointodyssey.com/) | Dynamic | Class B |
| [Spring](https://spring-benchmark.org/) | Dynamic | Class B |

Datasets are categorized into Class A, B, and C based on the preprocessing process of converting data format. 

After this step, all datasets will use our scripts (starts with `merge_` or `save_`) in `training/data/datasets/preprocess/` to
 save the metadata (camera parameters image paths, etc) to a single file to improve IO efficiency during training.

Some metadata saving scripts (such as the ones for hypersim, tartanair, tartanground) also include additional processing, such as filtering out the low quality data. 

### Class A
`Class A` needs to be first preprocessed with scripts provided by [VGGT](https://github.com/facebookresearch/vggt/tree/main/training). 
After that, you can run our scripts in `training/data/datasets/preprocess/` to save the metadata to improve IO efficiency.


### Class B
`Class B` needs to be preprocessed with scripts provided by [CUT3R](https://github.com/CUT3R/CUT3R/blob/main/docs/preprocess.md). After that, you can run our scripts in `training/data/datasets/preprocess/` to save the metadata to improve IO efficiency.


### Class C
`Class C` needs to be preprocessed with our custom scripts in `training/data/datasets/preprocess/`，starting with `preprocess_`. After that, you can run our scripts in `training/data/datasets/preprocess/` to save the metadata to improve IO efficiency.

I have provided some auxiliary scripts, starting with `download_`, `decompress_`, or `extract_` to help with the downloading, decompressing, and extracting of the datasets. Please check the comments in the scripts for more details.

