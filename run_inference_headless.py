import os
import time
import glob
import queue
import argparse
import collections
import numpy as np
import torch
from tqdm import tqdm

from zipmap.models.ZipMap_AR import ZipMap
from zipmap.utils.load_fn import load_and_preprocess_images
from zipmap.utils.pose_enc import pose_encoding_to_extri_intri
from zipmap.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3
from zipmap.utils.streaming_utils import ImageLoaderWorker, ResultSaverWorker

def run_inference(
    input_dir, 
    output_dir, 
    ckpt_path, 
    batch_size=1, 
    window_size=1, 
    use_ema=True, 
    align_first_view=True,
    num_load_threads=2,
    num_save_threads=4
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Model
    print(f"Loading model from {ckpt_path}...")
    model_config = {
        "img_size": 518,
        "patch_size": 14,
        "embed_dim": 1024,
        "enable_camera": False,
        "enable_camera_mlp": True,
        "enable_local_point": True,
        "enable_depth": True,
        "ttt_config": {
            "ttt_mode": True,
            "params": {
                "bias": True,
                "head_dim": 1024,
                "inter_multi": 2,
                "base_lr": 0.01,
                "muon_update_steps": 5,
                "use_gate_fn": True
            }
        }
    }
    model = ZipMap(**model_config)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    
    if use_ema and "ema" in checkpoint:
        print("Using EMA weights.")
        model_state_dict = checkpoint["ema"]
    else:
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    
    model.load_state_dict(model_state_dict, strict=False)
    model.eval()
    model = model.to(device)

    # 2. Prepare Data
    image_names = sorted(glob.glob(os.path.join(input_dir, "*")))
    if not image_names:
        print(f"No images found in {input_dir}")
        return
    print(f"Found {len(image_names)} images.")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)

    # 3. Setup Queues and Workers
    path_queue = queue.Queue()
    for p in image_names:
        path_queue.put(p)
    for _ in range(num_load_threads):
        path_queue.put(None) # poison pills

    image_queue = queue.Queue(maxsize=100)
    save_queue = queue.Queue(maxsize=100)

    loaders = [ImageLoaderWorker(path_queue, image_queue, name=f"Loader-{i}") for i in range(num_load_threads)]
    savers = [ResultSaverWorker(save_queue, name=f"Saver-{i}") for i in range(num_save_threads)]

    for w in loaders + savers:
        w.start()

    # 4. Inference Loop
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    processed_count = 0
    first_cam_inv = None
    current_state_list = None
    
    pbar = tqdm(total=len(image_names), desc="Inference")

    try:
        while processed_count < len(image_names):
            batch_data = []
            for _ in range(batch_size):
                if processed_count >= len(image_names):
                    break
                try:
                    data = image_queue.get(timeout=10)
                    batch_data.append(data)
                    processed_count += 1
                except queue.Empty:
                    break
            
            if not batch_data:
                continue

            # GPU Forward
            batch_tensors = [d["tensor"] for d in batch_data]
            images = torch.cat([t.unsqueeze(1) for t in batch_tensors], dim=1).to(device)

            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=dtype):
                    predictions = model(images, store_state=True, window_size=window_size, state_list=current_state_list)
                    current_state_list = predictions["state_list"]
                
                # Post-processing
                extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
                
                if predictions["depth"].ndim == 5:
                    predictions["depth"] = predictions["depth"].squeeze(-1)

                if align_first_view:
                    B, S = extrinsic.shape[:2]
                    extrinsic_homog = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(B, S, 1, 1).to(extrinsic.dtype)
                    extrinsic_homog[:, :, :3, :] = extrinsic
                    if first_cam_inv is None:
                        first_cam_inv = closed_form_inverse_se3(extrinsic_homog[:, 0])
                    new_extrinsic_homog = torch.matmul(extrinsic_homog, first_cam_inv.unsqueeze(1))
                    extrinsic = new_extrinsic_homog[:, :, :3, :]

                world_points_from_depth = unproject_depth_map_to_point_map(predictions["depth"], extrinsic, intrinsic)

            # Queuing for Save
            def to_numpy_and_remove_batch(tensor):
                if tensor is None: return None
                res = tensor.cpu().float().numpy()
                if res.shape[0] == 1:
                    return res[0]
                return res

            extrinsic_np = to_numpy_and_remove_batch(extrinsic)
            intrinsic_np = to_numpy_and_remove_batch(intrinsic)
            depth_np = to_numpy_and_remove_batch(predictions["depth"])
            wp_depth_np = to_numpy_and_remove_batch(world_points_from_depth)

            for i in range(len(batch_data)):
                img_path = batch_data[i]["path"]
                res = {
                    "extrinsic": extrinsic_np[i],
                    "intrinsic": intrinsic_np[i],
                    "depth": depth_np[i],
                    "world_points_from_depth": wp_depth_np[i],
                }
                save_path = os.path.join(output_dir, "predictions", f"{os.path.basename(img_path)}.npz")
                save_queue.put({"predictions": res, "save_path": save_path})
                image_queue.task_done()
            
            pbar.update(len(batch_data))

    finally:
        pbar.close()
        for _ in range(num_save_threads):
            save_queue.put(None)
        for w in loaders + savers:
            w.join()
    
    print(f"Finished! Results saved to {output_dir}/predictions")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless ZipMap Inference")
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="Path to images")
    parser.add_argument("--output_dir", "-o", type=str, default="./output_headless", help="Output directory")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--batch_size", "-b", type=int, default=1, help="Batch size for TTT")
    parser.add_argument("--window_size", "-w", type=int, default=1, help="TTT window size")
    parser.add_argument("--no_ema", action="store_false", dest="ema", help="Do not use EMA weights")
    parser.add_argument("--no_align", action="store_false", dest="align", help="Do not align to first view")
    
    args = parser.parse_args()
    
    run_inference(
        args.input_dir,
        args.output_dir,
        args.ckpt_path,
        batch_size=args.batch_size,
        window_size=args.window_size,
        use_ema=args.ema,
        align_first_view=args.align,
    )
