import os
import glob
import numpy as np
import torch
import cv2
import viser
import viser.transforms as tf
import argparse
from tqdm import tqdm
import time
import onnxruntime

def download_skyseg():
    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        import requests
        url = "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"
        r = requests.get(url, allow_redirects=True)
        with open("skyseg.onnx", 'wb') as f:
            f.write(r.content)

def run_skyseg(onnx_session, image):
    # Simplified version of sky segmentation
    h, w = image.shape[:2]
    img_input = cv2.resize(image, (320, 320))
    img_input = img_input.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))[None]
    
    ort_inputs = {onnx_session.get_inputs()[0].name: img_input}
    ort_outs = onnx_session.run(None, ort_inputs)
    mask = ort_outs[0][0, 0]
    mask = cv2.resize(mask, (w, h))
    return mask > 0.5 # 1 for non-sky, 0 for sky

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", "-n", type=str, required=True, help="Path to predictions folder containing .npz files")
    parser.add_argument("--image_dir", "-i", type=str, required=True, help="Path to original images")
    parser.add_argument("--mask_dir", "-m", type=str, default=None, help="Path to save/load masks (default: npz_dir/../sky_masks)")
    parser.add_argument("--stride", "-s", type=int, default=1, help="Stride for frames to visualize")
    parser.add_argument("--max_points", type=int, default=5_000_000, help="Max points to show at once")
    args = parser.parse_args()

    if args.mask_dir is None:
        args.mask_dir = os.path.join(os.path.dirname(os.path.abspath(args.npz_dir)), "sky_masks")
    os.makedirs(args.mask_dir, exist_ok=True)

    # 1. Load data paths
    npz_paths = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
    if not npz_paths:
        print(f"No .npz files found in {args.npz_dir}")
        return
    
    # Subsample by stride
    npz_paths = npz_paths[::args.stride]
    print(f"Visualizing {len(npz_paths)} frames (stride={args.stride})")

    server = viser.ViserServer()
    # ZipMap uses OpenCV coords (+X: right, +Y: down, +Z: forward).
    # To make the scene appear upright, we set the up direction to -Y.
    try:
        server.scene.set_up_direction("-y")
    except Exception:
        # Fallback for very old viser versions
        server.scene.set_up_direction("y")

    # GUI Elements
    conf_slider = server.gui.add_slider("Conf Threshold", min=0.0, max=1.0, step=0.01, initial_value=0.5)
    mask_sky_checkbox = server.gui.add_checkbox("Mask Sky", initial_value=True)
    point_size_slider = server.gui.add_slider("Point Size", min=0.001, max=0.05, step=0.001, initial_value=0.01)
    update_button = server.gui.add_button("Update Visualization")

    # State
    data_cache = []
    onnx_session = None

    def get_onnx():
        nonlocal onnx_session
        if onnx_session is None:
            download_skyseg()
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in onnxruntime.get_available_providers() else ["CPUExecutionProvider"]
            onnx_session = onnxruntime.InferenceSession("skyseg.onnx", providers=providers)
        return onnx_session

    def load_all_data():
        nonlocal data_cache
        data_cache = []
        for path in tqdm(npz_paths, desc="Loading data"):
            data = np.load(path)
            # Extrinsic is W2C (3, 4) or (4, 4)
            w2c = np.eye(4)
            w2c[:3, :4] = data["extrinsic"]
            c2w = np.linalg.inv(w2c)
            
            # Match image
            # Case 1: path is frame_0001.npz -> img_name is frame_0001
            # Case 2: path is frame_0001.jpg.npz -> img_name is frame_0001.jpg
            full_name = os.path.basename(path).replace(".npz", "")
            
            # Try direct match first (for case 2)
            img_path_direct = os.path.join(args.image_dir, full_name)
            if os.path.exists(img_path_direct):
                img_path = img_path_direct
            else:
                # Try glob match (for case 1 or cases with different extensions)
                base_name = os.path.splitext(full_name)[0]
                img_matches = glob.glob(os.path.join(args.image_dir, full_name + "*")) + \
                              glob.glob(os.path.join(args.image_dir, base_name + ".*"))
                
                if not img_matches:
                    print(f"Warning: Could not find image for {full_name} in {args.image_dir}. Skipping frame.")
                    continue
                img_path = img_matches[0]
            
            data_cache.append({
                "c2w": c2w,
                "points": data["world_points_from_depth"], # (H, W, 3)
                "img_path": img_path,
                "mask_path": os.path.join(args.mask_dir, img_name + "_mask.png")
            })

    load_all_data()

    def update_view():
        conf_val = conf_slider.value
        use_mask = mask_sky_checkbox.value
        
        all_points = []
        all_colors = []
        
        server.scene.reset()

        # Draw Cameras and Collect Points
        for i, item in enumerate(tqdm(data_cache, desc="Processing frames")):
            c2w = item["c2w"]
            
            # Show camera
            server.scene.add_camera_frustum(
                f"/cameras/cam_{i}",
                fov=2 * np.arctan(320 / (2 * 500)), # Placeholder FOV
                aspect=1.0,
                scale=0.1,
                color=(255, 0, 0),
                wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
                position=c2w[:3, 3]
            )

            # Process points
            points = item["points"] # (H, W, 3)
            H, W, _ = points.shape
            
            # Load image for colors
            img = cv2.imread(item["img_path"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[0] != H or img.shape[1] != W:
                img = cv2.resize(img, (W, H))
            
            # Masking
            valid_mask = np.ones((H, W), dtype=bool)
            if use_mask:
                if os.path.exists(item["mask_path"]):
                    mask = cv2.imread(item["mask_path"], cv2.IMREAD_GRAYSCALE) > 128
                else:
                    mask = run_skyseg(get_onnx(), img)
                    cv2.imwrite(item["mask_path"], (mask * 255).astype(np.uint8))
                
                if mask.shape[0] != H or mask.shape[1] != W:
                    mask = cv2.resize(mask.astype(np.uint8), (W, H)) > 0
                valid_mask &= mask

            # Confidence (ZipMap usually stores confidence in a separate key, 
            # but if not present in NPZ, we skip this for now or assume points are valid)
            # In run_inference_headless.py, we didn't save conf, but we can use depth > 0
            valid_mask &= (points[..., 2] > 0) 

            sel_points = points[valid_mask].reshape(-1, 3)
            sel_colors = img[valid_mask].reshape(-1, 3)
            
            # Downsample per frame if needed to avoid OOM
            if sel_points.shape[0] > 10000:
                idx = np.random.choice(sel_points.shape[0], 10000, replace=False)
                sel_points = sel_points[idx]
                sel_colors = sel_colors[idx]

            all_points.append(sel_points)
            all_colors.append(sel_colors)

        # Concatenate and show
        if all_points:
            all_points = np.concatenate(all_points, axis=0)
            all_colors = np.concatenate(all_colors, axis=0)
            
            # Global limit
            if all_points.shape[0] > args.max_points:
                idx = np.random.choice(all_points.shape[0], args.max_points, replace=False)
                all_points = all_points[idx]
                all_colors = all_colors[idx]

            server.scene.add_point_cloud(
                "/points",
                points=all_points,
                colors=all_colors,
                point_size=point_size_slider.value
            )
            print(f"Rendering {all_points.shape[0]} points")

    update_button.on_click(lambda _: update_view())
    
    # Initial view
    update_view()

    print("Viser server running. Press Ctrl+C to stop.")
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
