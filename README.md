<div align="center">
<h1>ZipMap: Linear-Time Stateful 3D Reconstruction via Test-Time Training</h1>

<h2>CVPR 2026</h2>

<a href="https://arxiv.org/abs/2603.04385"><img src="https://img.shields.io/badge/arXiv-2603.04385-b31b1b" alt="arXiv"></a>
<a href="https://haian-jin.github.io/ZipMap/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<!-- <a href='placeholder'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a> -->



[Haian Jin](https://haian-jin.github.io/)<sup>1,2</sup>, [Rundi Wu](https://rundiwu.github.io/)<sup>1</sup>, [Tianyuan Zhang](https://tianyuanzhang.com/)<sup>3</sup>, [Ruiqi Gao](https://ruiqigao.github.io/)<sup>1</sup>, [Jonathan T. Barron](https://jonbarron.info/)<sup>1</sup>, [Noah Snavely](https://www.cs.cornell.edu/~snavely/)<sup>1,2</sup>, [Aleksander Holynski](https://holynski.org/)<sup>1</sup>

<sup>1</sup>Google DeepMind &nbsp; <sup>2</sup>Cornell University &nbsp; <sup>3</sup>MIT
</div>

```bibtex
@inproceedings{jin2026zipmap,
    title     = {{ZipMap}: Linear-Time Stateful 3D Reconstruction via Test-Time Training},
    author    = {Jin, Haian and Wu, Rundi and Zhang, Tianyuan and Gao, Ruiqi and Barron, Jonathan T. and Snavely, Noah and Holynski, Aleksander},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year      = {2026}
}
```

## 0. Clarification
This is a **reimplementation** of the code for the paper "ZipMap: Linear-Time Stateful 3D Reconstruction via Test-Time Training".

We have verified that the re-implemented version matches the performance of the original. For any questions or issues, please contact Haian Jin at haianjin0415@gmail.com.

## 1. Environment Setup

```bash
conda create -n zipmap python=3.11
conda activate zipmap
pip install -e .
```

## 2. Inference
### 2.1 ZipMap Checkpoints

| Model | Description |
|-------|-------------|
| [**ZipMap**](YOUR_LINK_HERE) | Main model; no reference view specification (stage 3 checkpoint) |
| [ZipMap w/ reference view](YOUR_LINK_HERE) | With reference view specification (stage 2 checkpoint) |
<!-- | [ZipMap Streaming](YOUR_LINK_HERE) | Supports online/streaming inference (fine-tuned from ZipMap) |
| [ZipMap w/ state query](YOUR_LINK_HERE) | Supports state query (fine-tuned from ZipMap w/ reference view) | -->

### 2.2 Interactive Gradio Demo
Launch the demo locally:

```bash
python demo_gradio_zipmap.py --ckpt_path /path/to/your/checkpoint.pt
```
If using the checkpoint with reference view specification, you disable the affine invariant by setting `--affine_invariant false` when launching the demo.

### 2.2 Quantitative Evaluation
[**TODO**] See branch `evaluation` for code and instructions on how to run the quantitative evaluation.

## 3. Training
We train our model with FSDP (Fully Sharded Data Parallel) using GPUs with 80GB memory. If you have access to such hardware, you can run the training with the provided configs. If not, you can modify the configs to fit your hardware (e.g., by reducing batch size and using more aggressive FSDP sharding strategies).

### 3.1 Preparation
<details>
<summary><strong>Setup WandB for Logging</strong></summary>

Before training, fill in your WandB API key in `training/config/wandb_key.yaml` (get yours at [wandb.ai/authorize](https://wandb.ai/authorize)):

```yaml
wandb: YOUR_WANDB_API_KEY_HERE
```

</details>

<details>
<summary><strong>Download the Pretrained VGGT Checkpoint</strong></summary>

Download the VGGT checkpoint:

```bash
mkdir -p checkpoints
wget -O checkpoints/vggt_checkpoint.pt https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt
```

</details>

### 3.2 Debug Run
Before running the full training, we recommend doing a quick debug run to ensure everything is set up correctly. This run only uses two datasets.

<details>
<summary><strong>First download the two datasets for the debug run</strong></summary>

**1. Prepare VKITTI dataset**
- a. Use `training/data/datasets/preprocess/download_vkitti.sh` to download the VKITTI dataset. (This script is from VGGT's repository)
- b. Use our script `training/data/datasets/preprocess/save_vkitti_metadata.py` to save the metadata to improve IO efficiency.

**2. Prepare MVS-Synth dataset**
- a. First download the [MVS-Synth dataset](https://phuang17.github.io/DeepMVS/mvs-synth.html). Choose the 720p version.
- b. Then, preprocess the MVS-Synth dataset by running the script `training/data/datasets/preprocess/preprocess_mvs_synth.py` to convert data format. (This script is from CUT3R's repository)
- c. Run our script `training/data/datasets/preprocess/save_mvs_synth_metadata.py` to save the metadata to improve IO efficiency.

</details>

After that, you can use the `default_debug` config for this purpose:
```bash
torchrun --nproc_per_node=8 training/launch.py --config default_debug base_data_dir=/path/to/your/data
```

**The first few iterations may be slow due to torch.compile for TTT modules.** After that, the per-iteration time should be a few seconds. 

### 3.3 Full Run
The full training uses 29 datasets, including 23 static datasets and 6 dynamic datasets. Please prepare the datasets following the instructions in [training/data/README.md](training/data/README.md). After that, you can run the full training with the following commands:

```bash
# Stage 1: Train on 23 static datasets.
torchrun --nproc_per_node=8 training/launch.py --config default_stage1_hi_res_static base_data_dir=/path/to/your/data

# Stage 2: Train on all 29 datasets (23 static + 6 dynamic datasets).
torchrun --nproc_per_node=8 training/launch.py --config default_stage2_hi_res_dynamic base_data_dir=/path/to/your/data checkpoint.resume_checkpoint_path=/path/to/your/stage1_checkpoint.ckpt

# Stage 3: Remove the reference view specification and keep tuning on all 29 datasets.
torchrun --nproc_per_node=8 training/launch.py --config default_stage3_hi_res_dynamic_aff_inv base_data_dir=/path/to/your/data checkpoint.resume_checkpoint_path=/path/to/your/stage2_checkpoint.ckpt
```

## Acknowledgements

Our code is built on top of the following repositories:
- [VGGT](https://github.com/facebookresearch/vggt)
- [CUT3R](https://github.com/CUT3R/CUT3R)
- [Pi3](https://github.com/yyfz/Pi3)
- [MoGe](https://github.com/microsoft/moge)
- [LaCT](https://github.com/a1600012888/LaCT)

We sincerely thank the authors of these repositories for their open-source contributions, which have greatly helped this project.

