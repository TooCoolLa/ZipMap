# Code for ZipMap (CVPR 2026); created by Haian Jin

model_config = {
    "img_size": 518,
    "patch_size": 14,
    "embed_dim": 1024,
    "enable_camera": False,
    "enable_camera_mlp": True,
    "enable_local_point": True,
    "enable_depth": True,
    "ttt_config": 
        {
            "ttt_mode": True,
            "params": {
                "bias": True,   
                "head_dim": 1024,
                "inter_multi": 2,
                "base_lr": 0.01,
                "muon_update_steps": 5,
                "use_gate_fn": True
            },
            "window_size": 1
        },
    "other_config": {
        "use_gradient_checkpointing_local_point": False,
        "use_gradient_checkpointing_depth": False,
        "affine_invariant": True, 
    }
}
BASE_CACHE_DIR = "./debug_testing/demo_cache/"

import os
import cv2
import torch
import numpy as np
import gradio as gr
import sys
import shutil
from datetime import datetime
import glob
import gc
import time
import argparse 
import queue
import collections
sys.path.append("zipmap/")

PAGE_SIZE = 30

def get_gallery_slice(all_paths, page):
    if not all_paths or len(all_paths) == 0:
        return None, "Page 0 of 0", 1
    
    total_pages = (len(all_paths) + PAGE_SIZE - 1) // PAGE_SIZE
    try:
        page = int(page)
    except (ValueError, TypeError):
        page = 1
        
    page = max(1, min(page, total_pages))
    
    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    
    slice_paths = all_paths[start:end]
    return slice_paths, f"Page {page} of {total_pages}", page

from visual_util import predictions_to_glb
from zipmap.models.ZipMap_AR import ZipMap
from zipmap.utils.load_fn import load_and_preprocess_images
from zipmap.utils.pose_enc import pose_encoding_to_extri_intri
from zipmap.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3, homogenize_points
from zipmap.utils.streaming_utils import ImageLoaderWorker, ResultSaverWorker

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing and loading ZipMap model...")

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, default="./checkpoints/checkpoint_online.pt", help='Path to the model checkpoint')
parser.add_argument('--ema', action='store_true', help='Use EMA weights if available')
parser.add_argument('--align_first_view', type=lambda x: x.lower() in ('true', '1', 'yes'), default=True, help='Align output point cloud to the first view coordinate system (default: True)')
parser.add_argument('--image_dir', '-i', type=str, default=None, help='Path to local image directory')
args = parser.parse_args()

model = ZipMap(**model_config)
ckpt_path = args.ckpt_path
checkpoint = torch.load(ckpt_path, map_location="cpu",weights_only=True)
if args.ema and "ema" in checkpoint:
    print("Using EMA weights from checkpoint.")
    model_state_dict = checkpoint["ema"] if "ema" in checkpoint else checkpoint
else:
    model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

missing, unexpected = model.load_state_dict(model_state_dict, strict=False)

if missing:
    print(f"Missing keys in state_dict: {missing}")
if unexpected:
    print(f"Unexpected keys in state_dict: {unexpected}")

model.eval()
model = model.to(device)


# -------------------------------------------------------------------------
# 1) Core model inference (Streaming)
# -------------------------------------------------------------------------
def run_model_streaming(target_dir, model, num_load_threads=2, num_save_threads=None, img_queue_size=200, save_queue_size=80, batch_size=1, window_size=1) -> dict:
    """
    Run the ZipMap model on images in the 'target_dir/images' folder using streaming multi-threading.
    """
    print(f"Streaming processing images from {target_dir} with batch_size={batch_size}, window_size={window_size}")
    if num_save_threads is None:
        num_save_threads = max(1, os.cpu_count() - 2)

    # Move model to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Get image paths
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    # Setup queues
    path_queue = queue.Queue()
    for p in image_names:
        path_queue.put(p)
    for _ in range(num_load_threads):
        path_queue.put(None)  # poison pills for loaders

    image_queue = queue.Queue(maxsize=img_queue_size)
    save_queue = queue.Queue(maxsize=save_queue_size)

    # Start workers
    loaders = [ImageLoaderWorker(path_queue, image_queue) for _ in range(num_load_threads)]
    savers = [ResultSaverWorker(save_queue) for _ in range(num_save_threads)]

    for w in loaders + savers:
        w.start()

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    first_cam_inv = None
    all_predictions = collections.defaultdict(list)
    current_state_list = None
    
    # Ensure predictions directory exists
    os.makedirs(os.path.join(target_dir, "predictions"), exist_ok=True)

    # Inference loop
    processed_count = 0
    while processed_count < len(image_names):
        batch_data = []
        # Collect up to batch_size frames
        for _ in range(batch_size):
            if processed_count >= len(image_names):
                break
            data = image_queue.get()
            if data is None: 
                break
            batch_data.append(data)
            processed_count += 1
        
        if not batch_data:
            break
        
        # Construct [1, S, 3, H, W] from list of [1, 3, H, W]
        batch_tensors = [d["tensor"] for d in batch_data]
        images = torch.cat([t.unsqueeze(1) for t in batch_tensors], dim=1).to(device)
        
        # Run inference
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                predictions = model(images, store_state=True, window_size=window_size, state_list=current_state_list)
                current_state_list = predictions["state_list"]
        
        # Post-processing: Pose (supports BxSx... format)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        
        # Convert tensors to numpy and remove batch dim (result: [S, ...])
        extrinsic_all = extrinsic.cpu().float().numpy().squeeze(0)
        intrinsic_all = intrinsic.cpu().float().numpy().squeeze(0)
        depth_all = predictions["depth"].cpu().float().numpy().squeeze(0)
        depth_conf_all = predictions["depth_conf"].cpu().float().numpy().squeeze(0) if "depth_conf" in predictions else [None] * len(batch_data)
        
        local_points_all = None
        local_points_conf_all = None
        if "local_points" in predictions:
            local_points_all = predictions["local_points"].cpu().float().numpy().squeeze(0)
            local_points_conf_all = predictions["local_points_conf"].cpu().float().numpy().squeeze(0) if "local_points_conf" in predictions else [None] * len(batch_data)

        # Extract results for each frame in the batch
        for i in range(len(batch_data)):
            image_path = batch_data[i]["path"]
            image_tensor_single = batch_data[i]["tensor"]
            
            curr_extrinsic = extrinsic_all[i]
            curr_intrinsic = intrinsic_all[i]
            curr_depth = depth_all[i]
            curr_depth_conf = depth_conf_all[i]

            if args.align_first_view:
                extrinsic_homog = np.eye(4)
                extrinsic_homog[:3, :] = curr_extrinsic
                if first_cam_inv is None:
                    first_cam_inv = closed_form_inverse_se3(extrinsic_homog[None])[0]
                
                new_extrinsic_homog = extrinsic_homog @ first_cam_inv
                curr_extrinsic = new_extrinsic_homog[:3, :]

            # World points from depth
            world_points_from_depth = unproject_depth_map_to_point_map(
                curr_depth[None], curr_extrinsic[None], curr_intrinsic[None]
            ).squeeze(0)
            
            res_to_save = {
                "extrinsic": curr_extrinsic,
                "intrinsic": curr_intrinsic,
                "depth": curr_depth,
                "depth_conf": curr_depth_conf,
                "world_points_from_depth": world_points_from_depth,
                "images": image_tensor_single.cpu().float().numpy().squeeze(0).transpose(1, 2, 0)
            }
            
            if local_points_all is not None:
                curr_local_points = local_points_all[i]
                curr_local_points_conf = local_points_conf_all[i]
                
                extrinsic_homog = np.eye(4)
                extrinsic_homog[:3, :] = curr_extrinsic
                cam_to_world = closed_form_inverse_se3(extrinsic_homog[None])[0]
                
                world_points = np.einsum('ij, hwj -> hwi', cam_to_world, homogenize_points(curr_local_points))
                res_to_save["world_points"] = world_points[..., :3]
                res_to_save["world_points_conf"] = curr_local_points_conf
                res_to_save["local_points"] = curr_local_points

            # Save individual frame
            save_path = os.path.join(target_dir, "predictions", f"{os.path.basename(image_path)}.npz")
            save_queue.put({"predictions": res_to_save, "save_path": save_path})
            
            # Accumulate for return (compatibility with existing visualization)
            for k, v in res_to_save.items():
                all_predictions[k].append(v)
            
            image_queue.task_done()
        
    # Cleanup
    for _ in range(num_save_threads):
        save_queue.put(None)
    for w in loaders + savers:
        w.join()
        
    # Stack accumulated results
    final_predictions = {}
    for k, v in all_predictions.items():
        final_predictions[k] = np.stack(v, axis=0)
    
    torch.cuda.empty_cache()
    return final_predictions


# -------------------------------------------------------------------------
# 2) Handle uploaded video/images --> produce target_dir + images
# -------------------------------------------------------------------------
def handle_uploads(input_video, input_images, target_fps=None):
    """
    Create a new 'target_dir' + 'images' subfolder, and place user-uploaded
    images or extracted frames from video into it. Return (target_dir, image_paths).

    Args:
        input_video: Uploaded video file
        input_images: Uploaded image files
        target_fps: Target frames per second for video sampling (default: 1.0)
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Create a unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = os.path.join(BASE_CACHE_DIR, f"input_images_{timestamp}")
    target_dir_images = os.path.join(target_dir, "images")

    # Clean up if somehow that folder already exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []

    # --- Handle images ---
    if input_images is not None:
        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    # --- Handle video ---
    if input_video is not None:
        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video

        vs = cv2.VideoCapture(video_path)
        video_fps = vs.get(cv2.CAP_PROP_FPS)

        # Use target_fps if specified, otherwise default to 1 FPS
        if target_fps is not None and target_fps > 0:
            fps_target = float(target_fps)
        else:
            fps_target = 1.0  # Default: 1 frame per second

        # Calculate frame interval based on target FPS
        # frame_interval = video_fps / target_fps
        # For example: 30fps video, target 2fps -> sample every 15 frames
        frame_interval = max(1, int(round(video_fps / fps_target)))

        print(f"Video FPS: {video_fps}, Target FPS: {fps_target}, Frame interval: {frame_interval}")

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1

    # Sort final images for gallery
    image_paths = sorted(image_paths)

    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


# -------------------------------------------------------------------------
# 3) Update gallery on upload
# -------------------------------------------------------------------------
def update_gallery_on_upload(input_video, input_images, target_fps):
    """
    Whenever user uploads or changes files, immediately handle them
    and show in the gallery. Return (target_dir, image_paths).
    If nothing is uploaded, returns "None" and empty list.
    """
    if not input_video and not input_images:
        return None, None, None, None
    target_dir, image_paths = handle_uploads(input_video, input_images, target_fps)
    return None, target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin 3D processing."


# -------------------------------------------------------------------------
# 4) Reconstruction: uses the target_dir plus any viz parameters
# -------------------------------------------------------------------------
def gradio_demo(
    target_dir,
    conf_thres=3.0,
    frame_filter="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    prediction_mode="Pointmap Regression",
    cam_size_factor=1.0,
    loading_threads=2,
    img_queue_size=200,
    save_threads=None,
    save_queue_size=80,
    batch_size=1,
    window_size=1,
):
    """
    Perform reconstruction using the already-created target_dir/images.
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please upload first.", None, None

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare frame_filter dropdown
    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    print("Running run_model_streaming...")
    with torch.no_grad():
        predictions = run_model_streaming(
            target_dir, 
            model, 
            num_load_threads=int(loading_threads), 
            num_save_threads=int(save_threads) if save_threads is not None else None,
            img_queue_size=int(img_queue_size),
            save_queue_size=int(save_queue_size),
            batch_size=int(batch_size),
            window_size=int(window_size)
        )

    # Handle None frame_filter
    if frame_filter is None:
        frame_filter = "All"

    # Build a GLB file name
    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}_camfactor{cam_size_factor}.glb",
    )

    # Convert predictions to GLB
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
        cam_size_factor=float(cam_size_factor),
    )

    glbscene.export(file_obj=glbfile)

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")
    log_msg = f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization."

    return glbfile, log_msg, gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True)


# -------------------------------------------------------------------------
# 5) Helper functions for UI resets + re-visualization
# -------------------------------------------------------------------------
def clear_fields():
    """
    Clears the 3D viewer, the stored target_dir, and empties the gallery.
    """
    return None


def update_log():
    """
    Display a quick log message while waiting.
    """
    return "Loading and Reconstructing..."


def update_visualization(
    target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode,
    cam_size_factor=1.0,
):
    """
    Reload saved predictions from npz, create (or reuse) the GLB for new parameters,
    and return it for the 3D viewer.
    """

    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, "No reconstruction available. Please click the Reconstruct button first."

    predictions_dir = os.path.join(target_dir, "predictions")
    if not os.path.exists(predictions_dir):
        return None, f"No predictions directory found at {predictions_dir}. Please run 'Reconstruct' first."

    # Decide which files to load
    if frame_filter == "All" or frame_filter is None:
        npz_files = sorted(glob.glob(os.path.join(predictions_dir, "*.npz")))
    else:
        # Extract filename from "5: image005.png"
        try:
            filename = frame_filter.split(": ", 1)[1]
            npz_files = [os.path.join(predictions_dir, f"{filename}.npz")]
        except (IndexError, ValueError):
            return None, f"Invalid frame filter format: {frame_filter}"

    if not npz_files:
        return None, f"No prediction files found in {predictions_dir} matching {frame_filter}."

    # Load and aggregate
    all_preds = collections.defaultdict(list)
    for f in npz_files:
        if not os.path.exists(f):
            continue
        data = np.load(f)
        for k in data.files:
            all_preds[k].append(data[k])
        data.close()

    if not all_preds:
        return None, "Failed to load any prediction data."

    # Stack along sequence dimension
    predictions = {k: np.stack(v, axis=0) for k, v in all_preds.items()}

    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}_camfactor{cam_size_factor}.glb",
    )

    if not os.path.exists(glbfile):
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=mask_sky,
            target_dir=target_dir,
            prediction_mode=prediction_mode,
            cam_size_factor=float(cam_size_factor),
        )
        glbscene.export(file_obj=glbfile)

    return glbfile, "Updating Visualization"


# -------------------------------------------------------------------------
# 6) Build Gradio UI
# -------------------------------------------------------------------------

# Initial values for Gradio components
initial_target_dir = "None"
initial_image_paths = None
initial_log_msg = "Please upload a video or images, then click Reconstruct."

if args.image_dir and os.path.isdir(args.image_dir):
    image_files = sorted([os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if image_files:
        print(f"Pre-loading {len(image_files)} images from {args.image_dir}")
        initial_target_dir, initial_image_paths = handle_uploads(None, image_files)
        initial_log_msg = f"Pre-loaded {len(image_files)} images from {args.image_dir}. Click 'Reconstruct' to begin."

initial_gallery_value, initial_page_info, initial_page_num = get_gallery_slice(initial_image_paths, 1)

theme = gr.themes.Ocean()
theme.set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
)

with gr.Blocks(
    theme=theme,
    css="""
    .custom-log * {
        font-style: italic;
        font-size: 22px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        font-weight: bold !important;
        color: transparent !important;
        text-align: center !important;
    }
    
    .example-log * {
        font-style: italic;
        font-size: 16px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent !important;
    }
    
    #my_radio .wrap {
        display: flex;
        flex-wrap: nowrap;
        justify-content: center;
        align-items: center;
    }

    #my_radio .wrap label {
        display: flex;
        width: 50%;
        justify-content: center;
        align-items: center;
        margin: 0;
        padding: 10px 0;
        box-sizing: border-box;
    }
    """,
) as demo:
    # Instead of gr.State, we use a hidden Textbox:
    num_images = gr.Textbox(label="num_images", visible=False, value="None")
    all_image_paths = gr.State(initial_image_paths)

    gr.HTML(
    """
    <h1>🤐 ZipMap Playground</h1>

    <div style="font-size: 16px; line-height: 1.5;">
    <p>Upload a video or a set of images to create a 3D reconstruction of a scene or object. Our model takes these images and generates a 3D point cloud, along with estimated camera poses.</p>

    <h3>Getting Started:</h3>
    <ol>
        <li><strong>Upload Your Data:</strong> Use the "Upload Video" or "Upload Images" buttons on the left to provide your input. For videos, you can adjust the "Video Sampling FPS" to control how many frames per second are extracted (default: 1 FPS).</li>
        <li><strong>Preview:</strong> Your uploaded images will appear in the gallery on the left.</li>
        <li><strong>Reconstruct:</strong> Click the "Reconstruct" button to start the 3D reconstruction process.</li>
        <li><strong>Visualize:</strong> The 3D reconstruction will appear in the viewer on the right. You can rotate, pan, and zoom to explore the model, and download the GLB file. Note the visualization of 3D points may be slow for a large number of input images.</li>
        <li>
        <strong>Adjust Visualization (Optional):</strong>
        After reconstruction, you can fine-tune the visualization using the options below
        <details style="display:inline;">
            <summary style="display:inline;">(<strong>click to expand</strong>):</summary>
            <ul>
            <li><em>Confidence Threshold:</em> Adjust the filtering of points based on confidence.</li>
            <li><em>Show Points from Frame:</em> Select specific frames to display in the point cloud.</li>
            <li><em>Camera Frustum Size:</em> Scale the size of the camera frustum visualizations.</li>
            <li><em>Show Camera:</em> Toggle the display of estimated camera positions.</li>
            <li><em>Filter Sky / Filter Black Background / Filter White Background:</em> Remove sky, black-background, or white-background points.</li>
            <li><em>Select a Prediction Mode:</em> Choose between "Depthmap and Camera Branch" or "Local Pointmap and Camera Branch."</li>
            </ul>
        </details>
        </li>
    </ol>
    <p><strong style="color: #0ea5e9;">Please note:</strong> <span style="color: #0ea5e9; font-weight: bold;">Our model typically reconstructs a scene in less than 1 second. However, visualizing 3D points may take tens of seconds due to third-party rendering, which are independent of our model's processing time. </span></p>
    </div>
    """
    )

    target_dir_output = gr.Textbox(label="Target Dir", visible=False, value=initial_target_dir)

    with gr.Row():
        with gr.Column(scale=2):
            input_video = gr.Video(label="Upload Video", interactive=True)
            input_images = gr.File(file_count="multiple", label="Upload Images", interactive=True)

            target_fps = gr.Number(
                label="Video Sampling FPS",
                value=1.0,
                minimum=0.1,
                maximum=30,
                step=0.5,
                info="Target frames per second for video sampling (default: 1.0 FPS) \n You need to re-upload the video after changing this value."
            )

            loading_threads = gr.Slider(
                label="Loading Threads",
                minimum=1,
                maximum=8,
                value=2,
                step=1
            )
            img_queue_size = gr.Slider(
                label="Image Queue Size",
                minimum=10,
                maximum=500,
                value=200,
                step=10
            )
            save_threads = gr.Slider(
                label="Saving Threads",
                minimum=1,
                maximum=os.cpu_count() or 8,
                value=max(1, (os.cpu_count() or 4) - 2),
                step=1
            )
            save_queue_size = gr.Slider(
                label="Saving Queue Size",
                minimum=10,
                maximum=200,
                value=80,
                step=10
            )

            with gr.Row():
                batch_size = gr.Slider(
                    label="Inference Batch Size",
                    minimum=1,
                    maximum=16,
                    value=1,
                    step=1
                )
                window_size = gr.Slider(
                    label="TTT Window Size",
                    minimum=1,
                    maximum=16,
                    value=1,
                    step=1
                )

            image_gallery = gr.Gallery(
                label="Preview",
                columns=4,
                height="300px",
                show_download_button=True,
                object_fit="contain",
                preview=True,
                value=initial_gallery_value
            )

            with gr.Row():
                prev_page_btn = gr.Button("Previous Page", size="sm")
                page_num = gr.Number(value=initial_page_num, label="Page", precision=0, minimum=1, step=1, scale=1)
                next_page_btn = gr.Button("Next Page", size="sm")
            
            page_info = gr.Markdown(initial_page_info)
        
        with gr.Column(scale=4):
            with gr.Column():
                gr.Markdown("**3D Reconstruction (Point Cloud and Camera Poses)**")
                log_output = gr.Markdown(
                    initial_log_msg, elem_classes=["custom-log"]
                )
                reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5, camera_position=(-90, 90, 3.0))

            with gr.Row():
                submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                clear_btn = gr.ClearButton(
                    [input_video, input_images, reconstruction_output, log_output, target_dir_output, image_gallery, all_image_paths, page_num],
                    scale=1,
                )

            with gr.Row():
                prediction_mode = gr.Radio(
                    ["Depthmap and Camera Branch", "Local Pointmap and Camera Branch"],
                    label="Select a Prediction Mode",
                    value="Depthmap and Camera Branch",
                    scale=1,
                    elem_id="my_radio",
                )

            with gr.Row():
                conf_thres = gr.Slider(minimum=0, maximum=100, value=50, step=0.1, label="Confidence Threshold (%)")
                frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")
                cam_size_factor = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Camera Frustum Size")
                with gr.Column():
                    show_cam = gr.Checkbox(label="Show Camera", value=True)
                    mask_sky = gr.Checkbox(label="Filter Sky", value=False)
                    mask_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
                    mask_white_bg = gr.Checkbox(label="Filter White Background", value=False)

    # -------------------------------------------------------------------------
    # Pagination Events
    # -------------------------------------------------------------------------
    prev_page_btn.click(
        fn=lambda paths, p: get_gallery_slice(paths, p - 1),
        inputs=[all_image_paths, page_num],
        outputs=[image_gallery, page_info, page_num]
    )
    next_page_btn.click(
        fn=lambda paths, p: get_gallery_slice(paths, p + 1),
        inputs=[all_image_paths, page_num],
        outputs=[image_gallery, page_info, page_num]
    )
    page_num.submit(
        fn=get_gallery_slice,
        inputs=[all_image_paths, page_num],
        outputs=[image_gallery, page_info, page_num]
    )

    # -------------------------------------------------------------------------
    # "Reconstruct" button logic:
    #  - Clear fields
    #  - Update log
    #  - gradio_demo(...) with the existing target_dir
    # -------------------------------------------------------------------------
    submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
        fn=update_log, inputs=[], outputs=[log_output]
    ).then(
        fn=gradio_demo,
        inputs=[
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            cam_size_factor,
            loading_threads,
            img_queue_size,
            save_threads,
            save_queue_size,
            batch_size,
            window_size,
        ],
        outputs=[reconstruction_output, log_output, frame_filter],
    )

    # -------------------------------------------------------------------------
    # Real-time Visualization Updates
    # -------------------------------------------------------------------------
    viz_inputs = [
        target_dir_output,
        conf_thres,
        frame_filter,
        mask_black_bg,
        mask_white_bg,
        show_cam,
        mask_sky,
        prediction_mode,
        cam_size_factor,
    ]
    viz_outputs = [reconstruction_output, log_output]

    conf_thres.change(update_visualization, viz_inputs, viz_outputs)
    frame_filter.change(update_visualization, viz_inputs, viz_outputs)
    mask_black_bg.change(update_visualization, viz_inputs, viz_outputs)
    mask_white_bg.change(update_visualization, viz_inputs, viz_outputs)
    show_cam.change(update_visualization, viz_inputs, viz_outputs)
    mask_sky.change(update_visualization, viz_inputs, viz_outputs)
    prediction_mode.change(update_visualization, viz_inputs, viz_outputs)
    cam_size_factor.change(update_visualization, viz_inputs, viz_outputs)

    # -------------------------------------------------------------------------
    # Auto-update gallery whenever user uploads or changes their files
    # -------------------------------------------------------------------------
    def wrap_update_gallery(input_video, input_images, target_fps):
        _, target_dir, all_paths, log_msg = update_gallery_on_upload(input_video, input_images, target_fps)
        gallery_slice, info, p_num = get_gallery_slice(all_paths, 1)
        return None, target_dir, all_paths, gallery_slice, info, p_num, log_msg

    upload_outputs = [reconstruction_output, target_dir_output, all_image_paths, image_gallery, page_info, page_num, log_output]

    input_video.change(
        fn=wrap_update_gallery,
        inputs=[input_video, input_images, target_fps],
        outputs=upload_outputs,
    )
    input_images.change(
        fn=wrap_update_gallery,
        inputs=[input_video, input_images, target_fps],
        outputs=upload_outputs,
    )
    target_fps.change(
        fn=wrap_update_gallery,
        inputs=[input_video, input_images, target_fps],
        outputs=upload_outputs,
    )

    demo.queue(max_size=20).launch(show_error=True, share=True)
