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
import threading
import queue

def download_skyseg():
    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        import requests
        url = "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"
        r = requests.get(url, allow_redirects=True)
        with open("skyseg.onnx", 'wb') as f:
            f.write(r.content)

def run_skyseg(onnx_session, image):
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
    parser.add_argument("--npz_dir", "-n", type=str, required=True, help="Path to predictions folder")
    parser.add_argument("--image_dir", "-i", type=str, required=True, help="Path to original images")
    parser.add_argument("--mask_dir", "-m", type=str, default=None, help="Path to masks")
    parser.add_argument("--stride", "-s", type=int, default=1, help="Stride for frames")
    parser.add_argument("--max_points_per_frame", type=int, default=5000, help="Max points per frame to show")
    args = parser.parse_args()

    if args.mask_dir is None:
        args.mask_dir = os.path.join(os.path.dirname(os.path.abspath(args.npz_dir)), "sky_masks")
    os.makedirs(args.mask_dir, exist_ok=True)

    npz_paths = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
    npz_paths = npz_paths[::args.stride]
    
    server = viser.ViserServer()
    try:
        server.scene.set_up_direction("-y")
    except Exception:
        server.scene.set_up_direction("y")

    # GUI Elements
    gui_status = server.gui.add_text("Status", initial_value="Ready", disabled=True)
    gui_progress = server.gui.add_text("Progress", initial_value="0/0", disabled=True)
    mask_sky_checkbox = server.gui.add_checkbox("Mask Sky", initial_value=True)
    point_size_slider = server.gui.add_slider("Point Size", min=0.001, max=0.05, step=0.001, initial_value=0.01)
    
    btn_start = server.gui.add_button("Start/Resume Loading")
    btn_stop = server.gui.add_button("Stop Loading")
    btn_clear = server.gui.add_button("Clear Scene")

    # Global State
    state = {
        "is_loading": False,
        "stop_requested": False,
        "onnx_session": None,
        "processed_count": 0
    }

    def get_onnx():
        if state["onnx_session"] is None:
            download_skyseg()
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in onnxruntime.get_available_providers() else ["CPUExecutionProvider"]
            state["onnx_session"] = onnxruntime.InferenceSession("skyseg.onnx", providers=providers)
        return state["onnx_session"]

    def loading_thread():
        state["is_loading"] = True
        gui_status.value = "Loading..."
        
        for i in range(state["processed_count"], len(npz_paths)):
            if state["stop_requested"]:
                break
            
            path = npz_paths[i]
            try:
                data = np.load(path)
                w2c = np.eye(4)
                w2c[:3, :4] = data["extrinsic"]
                c2w = np.linalg.inv(w2c)
                
                full_name = os.path.basename(path).replace(".npz", "")
                img_path_direct = os.path.join(args.image_dir, full_name)
                if os.path.exists(img_path_direct):
                    img_path = img_path_direct
                else:
                    base_name = os.path.splitext(full_name)[0]
                    img_matches = glob.glob(os.path.join(args.image_dir, full_name + "*")) + \
                                  glob.glob(os.path.join(args.image_dir, base_name + ".*"))
                    if not img_matches:
                        continue
                    img_path = img_matches[0]

                # Load image
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                points = data["world_points_from_depth"]
                H, W, _ = points.shape
                if img.shape[0] != H or img.shape[1] != W:
                    img = cv2.resize(img, (W, H))

                valid_mask = (points[..., 2] > 0)
                if mask_sky_checkbox.value:
                    mask_path = os.path.join(args.mask_dir, full_name + "_mask.png")
                    if os.path.exists(mask_path):
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 128
                    else:
                        mask = run_skyseg(get_onnx(), img)
                        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
                    
                    if mask.shape[0] != H or mask.shape[1] != W:
                        mask = cv2.resize(mask.astype(np.uint8), (W, H)) > 0
                    valid_mask &= mask

                sel_points = points[valid_mask].reshape(-1, 3)
                sel_colors = img[valid_mask].reshape(-1, 3)
                
                if sel_points.shape[0] > args.max_points_per_frame:
                    idx = np.random.choice(sel_points.shape[0], args.max_points_per_frame, replace=False)
                    sel_points = sel_points[idx]
                    sel_colors = sel_colors[idx]

                # Add to Viser
                server.scene.add_camera_frustum(
                    f"/cameras/cam_{i}",
                    fov=2 * np.arctan(320 / (2 * 500)),
                    aspect=1.0,
                    scale=0.05,
                    color=(255, 0, 0),
                    wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
                    position=c2w[:3, 3]
                )
                
                server.scene.add_point_cloud(
                    f"/points/frame_{i}",
                    points=sel_points,
                    colors=sel_colors,
                    point_size=point_size_slider.value
                )
                
                state["processed_count"] = i + 1
                gui_progress.value = f"{state['processed_count']}/{len(npz_paths)}"
                
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue

        state["is_loading"] = False
        state["stop_requested"] = False
        gui_status.value = "Finished" if state["processed_count"] == len(npz_paths) else "Stopped"

    @btn_start.on_click
    def _(_):
        if not state["is_loading"]:
            state["stop_requested"] = False
            threading.Thread(target=loading_thread, daemon=True).start()

    @btn_stop.on_click
    def _(_):
        state["stop_requested"] = True

    @btn_clear.on_click
    def _(_):
        state["stop_requested"] = True
        while state["is_loading"]:
            time.sleep(0.1)
        server.scene.reset()
        state["processed_count"] = 0
        gui_progress.value = "0/0"
        gui_status.value = "Ready"

    print(f"Viser server running at http://localhost:8080. Total frames: {len(npz_paths)}")
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
