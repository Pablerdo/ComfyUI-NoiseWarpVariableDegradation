import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

import comfy.model_management as mm
from comfy.utils import ProgressBar

script_directory = os.path.dirname(os.path.abspath(__file__))

from .noisewarp.noise_warp import NoiseWarper, mix_new_noise
from .noisewarp.raft import RaftOpticalFlow

def get_downtemp_noise(noise, noise_downtemp_interp, interp_to=13):   
    if noise_downtemp_interp == 'nearest':
        return resize_list(noise, interp_to)
    elif noise_downtemp_interp == 'blend':
        return downsamp_mean(noise, interp_to)
    elif noise_downtemp_interp == 'blend_norm':
        return normalized_noises(downsamp_mean(noise, interp_to))
    elif noise_downtemp_interp == 'randn':
        return torch.randn_like(resize_list(noise, interp_to))
    else:
        return noise

def downsamp_mean(x, l=13):
    return torch.stack([sum(u) / len(u) for u in split_into_n_sublists(x, l)])

def normalized_noises(noises):
    #Noises is in TCHW form
    return torch.stack([x / x.std(1, keepdim=True) for x in noises])

def resize_list(array:list, length: int):
    assert isinstance(length, int), "Length must be an integer, but got %s instead"%repr(type(length))
    assert length >= 0, "Length must be a non-negative integer, but got %i instead"%length

    if len(array) > 1 and length > 1:
        step = (len(array) - 1) / (length - 1)
    else:
        step = 0  # default step size to 0 if array has only 1 element or target length is 1
        
    indices = [round(i * step) for i in range(length)]
    
    if isinstance(array, np.ndarray) or isinstance(array, torch.Tensor):
        return array[indices]
    else:
        return [array[i] for i in indices]
    
def split_into_n_sublists(l, n):
    if n <= 0:
        raise ValueError("n must be greater than 0 but n is "+str(n))

    if isinstance(l, str):
        return ''.join(split_into_n_sublists(list(l), n))

    L = len(l)
    indices = [int(i * L / n) for i in range(n + 1)]
    return [l[indices[i]:indices[i + 1]] for i in range(n)]

def optical_flow_to_image(dx, dy, *, mode='saturation', sensitivity=None):
    """
    Visualize optical flow as an RGB image - and return the image.
    
    The hue represents the angle of the flow, while magnitude is represented by either brightness or saturation.
    It has the same general idea as torchvision.utils.flow_to_image - when mode = 'saturation'
    
    Args:
       dx (numpy.ndarray matrix): The x-component of the optical flow.
       dy (numpy.ndarray matrix): The y-component of the optical flow.
       mode (str, optional): The visualization mode. Can be:
           - 'saturation': The saturation represents the magnitude. Default.
           - 'brightness': The brightness represents the magnitude.
       sensitivity (float, optional): If not specified, flow magnitudes are normalized at every frame
           Otherwise, the flow magnitudes are multiplied by this amount before visualizing
           (If you expect a max magnitude of 5 for example, you should set sensitivity=1/5)

       TODO: Use floating-point HSV precision, and a custom mag factor
       mag_factor (float, optional): the magnitude will be scaled by this number if specified,
                                     otherwise the magnitude will be scaled with full_range
    
    Returns:
       numpy.ndarray: The RGB image visualizing the optical flow.
    
    Raises:
       AssertionError: If dx and dy are not float matrices with the same shape, or if the mode is invalid.

    EXAMPLE:
        (see get_optical_flow_via_pyflow's docstring for an example)
    """
    dx=dx.astype(float) # np.float16 doesnt work
    dy=dy.astype(float) # np.float16 doesnt work
    
    import cv2
    
    hsv = np.zeros((*dx.shape, 3), dtype=np.uint8)
    hsv[:] = 255
    mag, ang = cv2.cartToPolar(dx, dy)

    if sensitivity is None:
        norm_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    else:
        norm_mag = mag
        norm_mag = sensitivity * mag
        norm_mag = np.tanh(norm_mag) #Soft clip it between 0 and 1
        norm_mag = np.clip(norm_mag * 255, 0, 255)
        norm_mag = norm_mag.astype(np.uint8)       


    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., {'brightness': 2, 'saturation': 1}[mode]] = norm_mag       
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return rgb

class WarpedNoiseBase:
    RETURN_TYPES = ("LATENT", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("noise", "visualization", "optical_flows")
    FUNCTION = "warp"
    CATEGORY = "NoiseWarp"

    def _process_video_frames(self, images, noise_channels, device, downscale_factor, resize_flow, return_flows=True):
        B, H, W, C = images.shape
        video_frames = images.permute(0, 3, 1, 2)
        
        warper = NoiseWarper(
            c=noise_channels,
            h=resize_flow * H,
            w=resize_flow * W,
            device=device,
            post_noise_alpha=0,
            progressive_noise_alpha=0,
        )

        raft_model = RaftOpticalFlow(device, "large")
        raft_model.model.to(device)

        return self._compute_warped_noise(video_frames, warper, raft_model, downscale_factor, return_flows=return_flows)

    def _compute_warped_noise(self, video_frames, warper, raft_model, downscale_factor, return_flows=False):
        prev_video_frame = video_frames[0]
        noise = warper.noise
        down_noise = self._downscale_noise(noise, downscale_factor)
        numpy_noise = down_noise.cpu().numpy().astype(np.float16)

        numpy_noises = [numpy_noise]
        rgb_flows = []

        pbar = ProgressBar(len(video_frames) - 1)

        for video_frame in tqdm(video_frames[1:], desc="Calculating noise warp"):
            dx, dy = raft_model(prev_video_frame, video_frame)
            if return_flows:
                flow_rgb = optical_flow_to_image(dx.cpu().numpy(), dy.cpu().numpy(), mode='saturation', sensitivity=1)
                rgb_flows.append(flow_rgb)
            noise = warper(dx, dy).noise
            prev_video_frame = video_frame
            down_noise = self._downscale_noise(noise, downscale_factor)
            numpy_noises.append(down_noise.cpu().numpy().astype(np.float16))
            pbar.update(1)

        return np.stack(numpy_noises), np.stack(rgb_flows) if return_flows else None

    def warp(self, images, noise_channels, noise_downtemp_interp, degradation, 
             target_latent_count, latent_shape, spatial_downscale_factor, seed, model=None, sigmas=None, return_flows=True):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        torch.manual_seed(seed)
        
        resize_flow = 1
        resize_frames = 1
        downscale_factor = round(resize_frames * resize_flow) * spatial_downscale_factor

        numpy_noises, rgb_flows = self._process_video_frames(
            images, noise_channels, device, downscale_factor, resize_flow, return_flows=return_flows
        )

        # Process noise tensor
        noise_tensor = torch.from_numpy(numpy_noises).squeeze(1).cpu().float()
        downtemp_noise_tensor = get_downtemp_noise(
            noise_tensor,
            noise_downtemp_interp=noise_downtemp_interp,
            interp_to=target_latent_count,
        )
        
        downtemp_noise_tensor = downtemp_noise_tensor[None]
        downtemp_noise_tensor = mix_new_noise(downtemp_noise_tensor, degradation)
        print(downtemp_noise_tensor.shape)

        # Process visualization tensors
        vis_tensor_noises = downtemp_noise_tensor
        vis_tensor_noises = vis_tensor_noises[:, :, :min(noise_channels, 3), :, :]      
        vis_tensor_noises = vis_tensor_noises.squeeze(0).permute(0, 2, 3, 1).cpu().float()
        vis_tensor_noises = (vis_tensor_noises - vis_tensor_noises.min()) / (vis_tensor_noises.max() - vis_tensor_noises.min())
        if return_flows:
            vis_tensor_flows = torch.from_numpy(rgb_flows) / 255
        else:
            vis_tensor_flows = None

        if latent_shape == "BTCHW":
            downtemp_noise_tensor = downtemp_noise_tensor.permute(0, 2, 1, 3, 4)
        elif latent_shape == "BCHW":
            downtemp_noise_tensor = downtemp_noise_tensor.squeeze(0)

        if sigmas is not None:
            sigma = sigmas[0] - sigmas[-1]
            sigma /= model.model.latent_format.scale_factor
            downtemp_noise_tensor *= sigma

        return {"samples":downtemp_noise_tensor}, vis_tensor_noises, vis_tensor_flows

    @staticmethod
    def _downscale_noise(noise, downscale_factor):
        down_noise = F.interpolate(noise, scale_factor=1/downscale_factor, mode='area')
        return down_noise * downscale_factor

class GetWarpedNoiseFromVideo(WarpedNoiseBase):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input images to be warped"}),
                "noise_channels": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1}),
                "noise_downtemp_interp": (["nearest", "blend", "blend_norm", "randn", "disabled"], {"tooltip": "Interpolation method(s) for down-temporal noise"}),
                "target_latent_count": ("INT", {"default": 13, "min": 1, "max": 2048, "step": 1, "tooltip": "Interpolate to this many latent frames"}),
                "degradation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Degradation level(s) for the noise warp"}),
                "latent_shape": (["BTCHW", "BCTHW", "BCHW"], {"tooltip": "Shape of the output latent tensor, for example CogVideoX uses BCTHW, while HunYuanVideo uses BTCHW"}),
                "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "Optional, to get the latent scale factor"} ),
                "sigmas": ("SIGMAS", {"tooltip": "Optional, to scale the noise"}),
                "spatial_downscale_factor": ("INT", {"default": 8, "min": 1, "max": 1024, "step": 1, "tooltip": "latent space spatial scale factor"}),
            },
        }

class GetWarpedNoiseFromVideoAnimateDiff(WarpedNoiseBase):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input images to be warped"}),
                "degradation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Degradation level(s) for the noise warp"}),
                "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "model": ("MODEL", {"tooltip": "Optional, to get the latent scale factor"}),
                "sigmas": ("SIGMAS", {"tooltip": "Optional, to scale the noise"}),
            },
        }
    RETURN_TYPES = ("LATENT", "IMAGE",)
    RETURN_NAMES = ("noise", "visualization",)

    def warp(self, images, degradation, seed, model=None, sigmas=None):
        return super().warp(
            images=images,
            noise_channels=4,
            noise_downtemp_interp="disabled",
            degradation=degradation,
            target_latent_count=images.shape[0],
            latent_shape="BCHW",
            spatial_downscale_factor=8,
            seed=seed,
            model=model,
            sigmas=sigmas,
            return_flows=False
        )

class GetWarpedNoiseFromVideoCogVideoX(WarpedNoiseBase):
    @classmethod
    def INPUT_TYPES(s):
       return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input images to be warped"}),
                "noise_downtemp_interp": (["nearest", "blend", "blend_norm", "randn", "disabled"], {"tooltip": "Interpolation method(s) for down-temporal noise"}),
                "num_frames": ("INT", {"default": 49, "min": 1, "max": 2048, "step": 1, "tooltip": "Interpolate to this many frames"}),
                "degradation": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Degradation level(s) for the noise warp"}),
                "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            },
        }

    def warp(self, images, degradation, seed, noise_downtemp_interp, num_frames, model=None, sigmas=None):
        latent_frames = (num_frames - 1) // 4 + 1
        return super().warp(
            images=images,
            noise_channels=16,
            noise_downtemp_interp=noise_downtemp_interp,
            degradation=degradation,
            target_latent_count=latent_frames,
            latent_shape="BCTHW",
            spatial_downscale_factor=8,
            seed=seed,
            model=model,
            sigmas=sigmas
        )
    
class GetWarpedNoiseFromVideoHunyuan(WarpedNoiseBase):
    @classmethod
    def INPUT_TYPES(s):
       return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input images to be warped"}),
                "noise_downtemp_interp": (["nearest", "blend", "blend_norm", "randn", "disabled"], {"tooltip": "Interpolation method(s) for down-temporal noise"}),
                "num_frames": ("INT", {"default": 49, "min": 1, "max": 2048, "step": 1, "tooltip": "Interpolate to this many frames"}),
                "degradation": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Degradation level(s) for the noise warp"}),
                "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            },
        }
    RETURN_TYPES = ("LATENT", "IMAGE",)
    RETURN_NAMES = ("noise", "visualization",)

    def warp(self, images, degradation, seed, noise_downtemp_interp, num_frames, model=None, sigmas=None):
        latent_frames = (num_frames - 1) // 4 + 1
        return super().warp(
            images=images,
            noise_channels=16,
            noise_downtemp_interp=noise_downtemp_interp,
            degradation=degradation,
            target_latent_count=latent_frames,
            latent_shape="BTCHW",
            spatial_downscale_factor=8,
            seed=seed,
            model=model,
            sigmas=sigmas,
            return_flows=False
        )


NODE_CLASS_MAPPINGS = {
    "GetWarpedNoiseFromVideo": GetWarpedNoiseFromVideo,
    "GetWarpedNoiseFromVideoAnimateDiff": GetWarpedNoiseFromVideoAnimateDiff,
    "GetWarpedNoiseFromVideoCogVideoX": GetWarpedNoiseFromVideoCogVideoX,
    "GetWarpedNoiseFromVideoHunyuan": GetWarpedNoiseFromVideoHunyuan,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "GetWarpedNoiseFromVideo": "GetWarpedNoiseFromVideo",
    "GetWarpedNoiseFromVideoAnimateDiff": "GetWarpedNoiseFromVideoAnimateDiff",
    "GetWarpedNoiseFromVideoCogVideoX": "GetWarpedNoiseFromVideoCogVideoX",
    "GetWarpedNoiseFromVideoHunyuan": "GetWarpedNoiseFromVideoHunyuan",
    }
