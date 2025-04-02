import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

import comfy.model_management as mm
from comfy.utils import ProgressBar

script_directory = os.path.dirname(os.path.abspath(__file__))

from .noisewarp.noise_warp import NoiseWarper, double_mask_border_region, compute_alpha_map_2levels
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
    CATEGORY = "NoiseWarpVariableDegradation"

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
    
    def _blend_noise_with_alpha_tensor(self, noise_background, noise_foreground, alpha_map):
        """
        Apply variance-preserving noise blending with a spatially varying alpha map.
        
        Args:
            noise_background: Background noise tensor to blend
            noise_foreground: Foreground noise tensor to blend 
            alpha_map: Tensor containing alpha values (0-1) for each spatial position
            
        Returns:
            Blended noise tensor with the same shape as the inputs
        """
        # Ensure alpha_map has proper broadcasting dimensions
        while len(alpha_map.shape) < len(noise_background.shape):
            alpha_map = alpha_map.unsqueeze(1)
        
        # Apply variance-preserving blend formula
        numerator = noise_foreground * alpha_map + noise_background * (1 - alpha_map)
        denominator = (alpha_map ** 2 + (1 - alpha_map) ** 2) ** 0.5
        
        # Avoid division by zero
        denominator = torch.clamp(denominator, min=1e-6)
        
        # Return the blended result
        return numerator / denominator

    def _apply_spatial_degradation_to_warped_noise(self, warped_noise, alpha_map, upscale_factor=8):
        """
        Apply spatial degradation to warped noise by upscaling, applying alpha map, and downscaling
        
        Args:
            warped_noise: The warped noise tensor
            alpha_map: The degradation map (0 = keep original, 1 = full degradation)
            upscale_factor: How much to upscale for applying the degradation
        
        Returns:
            Modified warped noise with spatial degradation applied
        """

        # TODO: make the upscale factor something that is derived from the difference in the spatial dimensions of the warped_noise and the alpha_map

        # Get original shape
        original_shape = warped_noise.shape
        
        # Reshape if needed for interpolation (e.g., from BTCHW to BCHW)
        if len(original_shape) == 5:  # BTCHW format
            b, t, c, h, w = original_shape
            reshaped_noise = warped_noise.reshape(b*t, c, h, w)
        else:
            reshaped_noise = warped_noise
        
        # 1. Upscale the noise to "pixel space"
        upscaled_noise = F.interpolate(
            reshaped_noise,
            scale_factor=upscale_factor,
            mode='bilinear'
        )
        
        # 2. Create random noise of the same shape
        random_noise = torch.randn_like(upscaled_noise)
        
        print(f"original alpha_map shape: {alpha_map.shape}")

        alpha_map_resized = alpha_map
        # 3. Resize alpha_map to match upscaled noise spatial dimensions
        if alpha_map.shape[-2:] != upscaled_noise.shape[-2:]:
            # Ensure alpha_map has proper batch/channel dims for interpolation
            if len(alpha_map.shape) < len(upscaled_noise.shape):
                alpha_map = alpha_map.unsqueeze(1)  # Add channel dim if needed
                
            alpha_map_resized = F.interpolate(
                alpha_map,
                size=upscaled_noise.shape[-2:],
                mode='bilinear'
            )
        else:
            print("did not resize alpha_map")
            alpha_map_resized = alpha_map
            
        # Ensure alpha map has proper shape for broadcasting
        while len(alpha_map_resized.shape) < len(upscaled_noise.shape):
            alpha_map_resized = alpha_map_resized.unsqueeze(1)
        
        print(f"alpha_map_resized shape: {alpha_map_resized.shape}")
        print(f"upscaled_noise shape: {upscaled_noise.shape}")
        # 4. Apply the degradation by blending original and random noise

        degraded_noise = self._blend_noise_with_alpha_tensor(upscaled_noise, random_noise, alpha_map_resized)

        # degraded_noise = (
        #     upscaled_noise * (1 - alpha_map_resized) + 
        #     random_noise * alpha_map_resized
        # )
        
        # 5. Downscale back to original resolution
        downscaled_noise = F.interpolate(
            degraded_noise,
            size=reshaped_noise.shape[-2:],
            mode='area'
        )
        
        # 6. Reshape back to original format if needed
        if len(original_shape) == 5:  # BTCHW format
            downscaled_noise = downscaled_noise.reshape(original_shape)
        
        return downscaled_noise

    def warp(self, images, masks, noise_channels, noise_downtemp_interp, degradation, boundary_degradation, second_boundary_degradation,
             target_latent_count, latent_shape, spatial_downscale_factor, seed, model=None, sigmas=None, return_flows=True, output_device="CPU"):
        device = mm.get_torch_device()
        
        torch.manual_seed(seed)
        
        resize_flow = 1
        resize_frames = 1
        downscale_factor = round(resize_frames * resize_flow) * spatial_downscale_factor

        numpy_noises, rgb_flows = self._process_video_frames(
            images, noise_channels, device, downscale_factor, resize_flow, return_flows=return_flows
        )

        # Process noise tensor
        noise_tensor = torch.from_numpy(numpy_noises).squeeze(1).cpu().float()

        print(f"noise_tensor shape: {noise_tensor.shape}")

        # B, H, W, C = masks.shape

        # convert BHWC to BCHW
        mask_frames = masks.permute(0, 3, 1, 2)

        print(f"mask_frames shape: {mask_frames.shape}")


        # Create alpha map for variable degradation (moved from mix_new_noise_variable_degradation)
        # First, compute boundary masks in pixel space
        boundary_px1 = 10  # innermost boundary
        boundary_px2 = 20  # outer boundary
        boundary_mask, second_boundary_mask, mask_tensor = double_mask_border_region(mask_frames, boundary_px1, boundary_px2)

        # Then compute alpha map in pixel space
        pixel_alpha_map = compute_alpha_map_2levels(mask_tensor, boundary_mask, second_boundary_mask, 
                                                boundary_degradation, second_boundary_degradation, degradation)
        
        print(f"pixel_alpha_map shape: {pixel_alpha_map.shape}")
        print(f"pixel_alpha_map: {pixel_alpha_map[0]}")

        noise_tensor = self._apply_spatial_degradation_to_warped_noise(noise_tensor, pixel_alpha_map)

        downtemp_noise_tensor = get_downtemp_noise(
            noise_tensor,
            noise_downtemp_interp=noise_downtemp_interp,
            interp_to=target_latent_count,
        )
        
        downtemp_noise_tensor = downtemp_noise_tensor[None]

        print(f"downtemp_noise_tensor shape: {downtemp_noise_tensor.shape}")

        # Process visualization tensors
        vis_tensor_noises = downtemp_noise_tensor
        vis_tensor_noises = vis_tensor_noises[:, :, :min(noise_channels, 3), :, :]      
        vis_tensor_noises = vis_tensor_noises.squeeze(0).permute(0, 2, 3, 1).cpu().float()
        vis_tensor_noises = (vis_tensor_noises - vis_tensor_noises.min()) / (vis_tensor_noises.max() - vis_tensor_noises.min())
        target_len = images.shape[0]
        repeat_count = (target_len + vis_tensor_noises.shape[0] - 1) // vis_tensor_noises.shape[0]
        vis_tensor_noises = vis_tensor_noises.repeat_interleave(repeat_count, dim=0)[:target_len]
        if return_flows:
            vis_tensor_flows = torch.from_numpy(rgb_flows) / 255
        else:
            vis_tensor_flows = None

        if latent_shape == "BTCHW":
            downtemp_noise_tensor = downtemp_noise_tensor.permute(0, 2, 1, 3, 4)
        elif latent_shape == "BCHW":
            downtemp_noise_tensor = downtemp_noise_tensor.squeeze(0)

        if sigmas is not None:
            sigmas = sigmas.cpu()
            sigma = sigmas[0] - sigmas[-1]
            sigma /= model.model.latent_format.scale_factor
            downtemp_noise_tensor *= sigma

        downtemp_noise_tensor = downtemp_noise_tensor.to(device) if output_device == "GPU" else downtemp_noise_tensor.cpu()

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
                "binary_images": ("IMAGE", {"tooltip": "Input binary images to define the degradation boundaries"}),
                "noise_channels": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1}),
                "noise_downtemp_interp": (["nearest", "blend", "blend_norm", "randn", "disabled"], {"tooltip": "Interpolation method(s) for down-temporal noise"}),
                "target_latent_count": ("INT", {"default": 13, "min": 1, "max": 2048, "step": 1, "tooltip": "Interpolate to this many latent frames"}),
                "boundary_degradation": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Degradation level(s) for the noise warp at the boundaries"}),
                "second_boundary_degradation": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Degradation level(s) for the noise warp at the second boundary"}),
                "degradation": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Degradation level(s) for the noise warp at the inner region of the mask"}),
                "latent_shape": (["BTCHW", "BCTHW", "BCHW"], {"tooltip": "Shape of the output latent tensor, for example CogVideoX uses BCTHW, while HunYuanVideo uses BTCHW"}),
                "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "Optional, to get the latent scale factor"} ),
                "sigmas": ("SIGMAS", {"tooltip": "Optional, to scale the noise"}),
                "spatial_downscale_factor": ("INT", {"default": 8, "min": 1, "max": 1024, "step": 1, "tooltip": "latent space spatial scale factor"}),
                "output_device": (["GPU", "CPU"], {"default": "CPU", "tooltip": "Device to return the latents on"}),
            },
        }

class GetWarpedNoiseFromVideoAnimateDiff(WarpedNoiseBase):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input images to be warped"}),
                "binary_images": ("IMAGE", {"tooltip": "Input binary images to define the degradation boundaries"}),
                "degradation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Degradation level(s) for the noise warp"}),
                "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "model": ("MODEL", {"tooltip": "Optional, to get the latent scale factor"}),
                "sigmas": ("SIGMAS", {"tooltip": "Optional, to scale the noise"}),
                "output_device": (["GPU", "CPU"], {"default": "CPU", "tooltip": "Device to return the latents on"}),
            },
        }
    RETURN_TYPES = ("LATENT", "IMAGE",)
    RETURN_NAMES = ("noise", "visualization",)

    def warp(self, images, binary_images, degradation, seed, model=None, sigmas=None, output_device="CPU"):
        return super().warp(
            images=images,
            masks=binary_images,
            noise_channels=4,
            noise_downtemp_interp="disabled",
            degradation=degradation,
            target_latent_count=images.shape[0],
            latent_shape="BCHW",
            spatial_downscale_factor=8,
            seed=seed,
            model=model,
            sigmas=sigmas,
            return_flows=False,
            output_device=output_device
        )

class GetWarpedNoiseFromVideoCogVideoX(WarpedNoiseBase):
    @classmethod
    def INPUT_TYPES(s):
       return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input images to be warped"}),
                "binary_images": ("IMAGE", {"tooltip": "Input binary images to define the degradation boundaries"}),
                "noise_downtemp_interp": (["nearest", "blend", "blend_norm", "randn", "disabled"], {"tooltip": "Interpolation method(s) for down-temporal noise"}),
                "num_frames": ("INT", {"default": 49, "min": 1, "max": 2048, "step": 1, "tooltip": "Interpolate to this many frames"}),
                "degradation": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Degradation level(s) for the noise warp"}),
                "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "output_device": (["GPU", "CPU"], {"default": "CPU", "tooltip": "Device to return the latents on"}),
            },
        }

    def warp(self, images, binary_images, degradation, seed, noise_downtemp_interp, num_frames, model=None, sigmas=None, output_device="CPU"):
        latent_frames = (num_frames - 1) // 4 + 1
        return super().warp(
            images=images,
            masks=binary_images,
            noise_channels=16,
            noise_downtemp_interp=noise_downtemp_interp,
            degradation=degradation,
            target_latent_count=latent_frames,
            latent_shape="BCTHW",
            spatial_downscale_factor=8,
            seed=seed,
            model=model,
            sigmas=sigmas,
            output_device=output_device
        )
    
class GetWarpedNoiseFromVideoHunyuan(WarpedNoiseBase):
    @classmethod
    def INPUT_TYPES(s):
       return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input images to be warped"}),
                "binary_images": ("IMAGE", {"tooltip": "Input binary images to define the degradation boundaries"}),
                "noise_downtemp_interp": (["nearest", "blend", "blend_norm", "randn", "disabled"], {"tooltip": "Interpolation method(s) for down-temporal noise"}),
                "num_frames": ("INT", {"default": 49, "min": 1, "max": 2048, "step": 1, "tooltip": "Interpolate to this many frames"}),
                "boundary_degradation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Degradation level(s) for the noise warp at the boundaries"}),
                "second_boundary_degradation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Degradation level(s) for the noise warp at the second boundary"}),
                "degradation": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Degradation level(s) for the noise warp at the inner region of the mask"}),
                "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            },
        }
    RETURN_TYPES = ("LATENT", "IMAGE",)
    RETURN_NAMES = ("noise", "visualization",)

    def warp(self, images, binary_images, degradation, boundary_degradation, second_boundary_degradation, seed, noise_downtemp_interp, num_frames, model=None, sigmas=None):
        latent_frames = (num_frames - 1) // 4 + 1
        return super().warp(
            images=images,
            masks=binary_images,
            noise_channels=16,
            noise_downtemp_interp=noise_downtemp_interp,
            degradation=degradation,
            boundary_degradation=boundary_degradation,
            second_boundary_degradation=second_boundary_degradation,
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
