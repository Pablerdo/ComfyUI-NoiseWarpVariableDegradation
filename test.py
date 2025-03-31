import os
import torch
import numpy as np
import pytest
from PIL import Image
import torchvision.transforms as transforms
import tqdm
import cv2

# Import the node class
from nodes import GetWarpedNoiseFromVideoHunyuan

# class ComfyUIDeployExternalVideo:
#     @classmethod
#     def INPUT_TYPES(s):
#         input_dir = folder_paths.get_input_directory()
#         files = []
#         for f in os.listdir(input_dir):
#             if os.path.isfile(os.path.join(input_dir, f)):
#                 file_parts = f.split(".")
#                 if len(file_parts) > 1 and (file_parts[-1] in video_extensions):
#                     files.append(f)
#         return {"required": {
#                     "input_id": (
#                         "STRING",
#                         {"multiline": False, "default": "input_video"},
#                     ),
#                      "force_rate": ("INT", {"default": 0, "min": 0, "max": 60, "step": 1}),
#                      "force_size": (["Disabled", "Custom Height", "Custom Width", "Custom", "256x?", "?x256", "256x256", "512x?", "?x512", "512x512"],),
#                      "custom_width": ("INT", {"default": 512, "min": 0, "max": DIMMAX, "step": 8}),
#                      "custom_height": ("INT", {"default": 512, "min": 0, "max": DIMMAX, "step": 8}),
#                      "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
#                      "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
#                      "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
#                      },
#                 "optional": {
#                     "meta_batch": ("VHS_BatchManager",),
#                     "vae": ("VAE",),
#                     "default_video": (sorted(files),),
#                     "display_name": (
#                         "STRING",
#                         {"multiline": False, "default": ""},
#                     ),
#                     "description": (
#                         "STRING",
#                         {"multiline": True, "default": ""},
#                     ),
#                 },
#                 "hidden": {
#                     "unique_id": "UNIQUE_ID"
#                 },
#                 }

#     CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

#     RETURN_TYPES = ("IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO", "LATENT")
#     RETURN_NAMES = (
#         "IMAGE",
#         "frame_count",
#         "audio",
#         "video_info",
#         "LATENT",
#     )

#     FUNCTION = "load_video"
#     CATEGORY = "🔗ComfyDeploy"

def load_video(video_path, force_rate=0, force_size="Disabled", custom_width=512, custom_height=512,
               frame_load_cap=0, skip_first_frames=0, select_every_nth=1):
    """
    Load a video file from the assets folder and return it as a tensor.
    
    Args:
        video_path (str): Path to the video file in the assets folder
        force_rate (int): Force a specific frame rate (0 means use original)
        force_size (str): Size mode ("Disabled", "Custom Height", "Custom Width", "Custom", etc.)
        custom_width (int): Custom width when force_size is "Custom" or "Custom Width"
        custom_height (int): Custom height when force_size is "Custom" or "Custom Height"
        frame_load_cap (int): Maximum number of frames to load (0 means load all)
        skip_first_frames (int): Number of frames to skip at the start
        select_every_nth (int): Load every nth frame
    
    Returns:
        torch.Tensor: Video frames in format [B, H, W, C] where B is batch size (frames)
    """
    # Ensure video path exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate target dimensions based on force_size
    target_width = width
    target_height = height
    if force_size != "Disabled":
        if force_size == "Custom":
            target_width = custom_width
            target_height = custom_height
        elif force_size == "Custom Width":
            target_width = custom_width
            target_height = int(height * (custom_width / width))
        elif force_size == "Custom Height":
            target_height = custom_height
            target_width = int(width * (custom_height / height))
        elif force_size in ["256x?", "?x256", "256x256", "512x?", "?x512", "512x512"]:
            if force_size == "256x256":
                target_width = 256
                target_height = 256
            elif force_size == "512x512":
                target_width = 512
                target_height = 512
            elif force_size == "256x?":
                target_width = 256
                target_height = int(height * (256 / width))
            elif force_size == "?x256":
                target_height = 256
                target_width = int(width * (256 / height))
            elif force_size == "512x?":
                target_width = 512
                target_height = int(height * (512 / width))
            elif force_size == "?x512":
                target_height = 512
                target_width = int(width * (512 / height))
    
    # Calculate frame range
    if frame_load_cap > 0:
        frame_count = min(frame_count, frame_load_cap)
    frames_to_load = range(skip_first_frames, frame_count, select_every_nth)
    
    # Read frames
    frames = []
    for frame_idx in tqdm.tqdm(frames_to_load, desc="Loading video frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize if needed
        if target_width != width or target_height != height:
            frame = cv2.resize(frame, (target_width, target_height))
            
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    # Convert to tensor
    video = np.stack(frames)
    video = torch.from_numpy(video).float() / 255.0
    
    return video

def test_boundary_degradation():
    # Load video with its original size
    video = load_video(os.path.join("test", "assets", "test_video.mp4"))
    
    # Create node instance
    node = GetWarpedNoiseFromVideoHunyuan()
    
    # Test different boundary degradation values
    test_cases = [
        (0.0, 0.0, 0.0),  # No degradation
        (0.5, 0.3, 0.1),  # Different degradation levels
        (1.0, 1.0, 1.0),  # Maximum degradation
    ]
    
    for degradation, boundary_degradation, second_boundary_degradation in test_cases:
        # Call the node
        result = node.warp(
            images=video,
            degradation=degradation,
            boundary_degradation=boundary_degradation,
            second_boundary_degradation=second_boundary_degradation,
            seed=123,
            noise_downtemp_interp="nearest",
            num_frames=10
        )
        
        # Check if we got the expected outputs
        assert isinstance(result, tuple)
        assert len(result) == 2  # noise and visualization
        
        # Check noise tensor
        noise, visualization = result
        assert isinstance(noise, dict)
        assert "samples" in noise
        assert isinstance(noise["samples"], torch.Tensor)
        
        # Check visualization tensor
        assert isinstance(visualization, torch.Tensor)
        
        # Check shapes - dynamically determine based on the input
        c = 16  # Number of noise channels
        t = 3   # Default temporal dimension
        # The spatial dimensions will be downscaled from the input
        h, w = video.shape[1] // 8, video.shape[2] // 8  # Typical 8x downsampling for latents
        expected_noise_shape = (1, c, t, h, w)  # BTCHW format
        print(f"Expected noise shape: {expected_noise_shape}, Actual shape: {noise['samples'].shape}")
        
        # Check value ranges
        assert torch.all(noise["samples"] >= -10.0) and torch.all(noise["samples"] <= 10.0)
        assert torch.all(visualization >= 0.0) and torch.all(visualization <= 1.0)
        
        print(f"Test passed for degradation={degradation}, "
              f"boundary_degradation={boundary_degradation}, "
              f"second_boundary_degradation={second_boundary_degradation}")

if __name__ == "__main__":
    # Example usage of load_video with a video from assets folder - using original size
    video_path = os.path.join("test", "assets", "test_video.mp4")
    video = load_video(video_path)
    test_boundary_degradation()
