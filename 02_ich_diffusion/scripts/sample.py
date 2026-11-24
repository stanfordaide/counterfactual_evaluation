# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import math
import os
import random
import time
from datetime import datetime
import warnings
import gc

import monai
import torch
from monai.data import MetaTensor
from monai.inferers.inferer import DiffusionInferer
from monai.transforms import Compose, SaveImage
from monai.utils import set_determinism
from tqdm import tqdm
from monai.inferers.inferer import SlidingWindowInferer
from monai.networks.schedulers import RFlowScheduler, DDPMScheduler

# from augmentation import augmentation
# from find_masks import find_masks
# from quality_check import is_outlier
from utils import (
    binarize_labels,
    general_mask_generation_post_process,
    get_body_region_index_from_mask,
    remap_labels,
    dynamic_infer,
)

import numpy as np


class ReconModel(torch.nn.Module):
    """
    A PyTorch module for reconstructing images from latent representations.

    Attributes:
        autoencoder: The autoencoder model used for decoding.
        scale_factor: Scaling factor applied to the input before decoding.
    """

    def __init__(self, autoencoder, scale_factor):
        super().__init__()
        self.autoencoder = autoencoder
        self.scale_factor = scale_factor

    def forward(self, z):
        """
        Decode the input latent representation to an image.

        Args:
            z (torch.Tensor): The input latent representation.

        Returns:
            torch.Tensor: The reconstructed image.
        """
        recon_pt_nda = self.autoencoder.decode_stage_2_outputs(z / self.scale_factor)
        return recon_pt_nda


def initialize_noise_latents(latent_shape, device):
    """
    Initialize random noise latents for image generation with float16.

    Args:
        latent_shape (tuple): The shape of the latent space.
        device (torch.device): The device to create the tensor on.

    Returns:
        torch.Tensor: Initialized noise latents.
    """
    return (
        torch.randn(
            [
                1,
            ]
            + list(latent_shape)
        )
        .half()
        .to(device)
    )



def crop_img_body_mask(synthetic_images, combine_label):
    """
    Crop the synthetic image using a body mask.

    Args:
        synthetic_images (torch.Tensor): The synthetic images.
        combine_label (torch.Tensor): The body mask.

    Returns:
        torch.Tensor: The cropped synthetic images.
    """
    synthetic_images[combine_label == 0] = -1000
    return synthetic_images



def revise_transform(data, method: str) -> np.ndarray:
    
    if method == "piecewise_linear_mapping":
        arr = data


        R_POINTS = np.array([-1000.0, -100.0,   0.0,  80.0, 300.0, 2000.0], dtype=np.float32) ## Input HU breakpoints,
        S_POINTS = np.array([-1000.0, -600.0, -200.0, 400.0, 700.0, 1000.0], dtype=np.float32)  ## Output values at those breakpoints.

        r_points = R_POINTS
        s_points = S_POINTS
        ## clip the arr to the min and max of r_points

        arr = np.clip(arr, s_points[0], s_points[-1])
        y = arr.astype(np.float32)

        x = np.zeros_like(y, dtype=np.float32)

        # Below first s: extrapolate
        below = y <= s_points[0]
        if np.any(below):
            a = (r_points[1] - r_points[0]) / (s_points[1] - s_points[0])
            x[below] = r_points[0] + a * (y[below] - s_points[0])

        # Above last s: extrapolate
        above = y >= s_points[-1]
        if np.any(above):
            a = (r_points[-1] - r_points[-2]) / (s_points[-1] - s_points[-2])
            x[above] = r_points[-1] + a * (y[above] - s_points[-1])

        # Middle segments
        for i in range(len(s_points) - 1):
            s0, s1 = s_points[i],   s_points[i + 1]
            r0, r1 = r_points[i],   r_points[i + 1]
            a = (r1 - r0) / (s1 - s0)

            mask = (y > s0) & (y < s1)
            x[mask] = r0 + a * (y[mask] - s0)

        # Exact knot points in warped space
        for s_k, r_k in zip(s_points, r_points):
            mask_eq = (y == s_k)
            x[mask_eq] = r_k

        x = x.astype(np.int16)
        data = x

    return data



def ldm_conditional_sample_one_image_with_piecewise_linear_postprocessing(
    autoencoder,
    diffusion_unet,
    controlnet,
    noise_scheduler,
    scale_factor,
    device,
    combine_label_or,
    spacing_tensor,
    latent_shape,
    output_size,
    noise_factor,
    top_region_index_tensor=None,
    bottom_region_index_tensor=None,
    context=None,
    modality_tensor=None,
    num_inference_steps=1000,
    autoencoder_sliding_window_infer_size=[96, 96, 96],
    autoencoder_sliding_window_infer_overlap=0.6667,
):
    """
    Generate a single synthetic image using a latent diffusion model with controlnet.

    Args:
        autoencoder (nn.Module): The autoencoder model.
        diffusion_unet (nn.Module): The diffusion U-Net model.
        controlnet (nn.Module): The controlnet model.
        noise_scheduler: The noise scheduler for the diffusion process.
        scale_factor (float): Scaling factor for the latent space.
        device (torch.device): The device to run the computation on.
        combine_label_or (torch.Tensor): The combined label tensor.
        spacing_tensor (torch.Tensor): Tensor specifying the spacing.
        latent_shape (tuple): The shape of the latent space.
        output_size (tuple): The desired output size of the image.
        noise_factor (float): Factor to scale the initial noise.
        top_region_index_tensor (torch.Tensor): Tensor specifying the top region index. Defaults to None.
        bottom_region_index_tensor (torch.Tensor): Tensor specifying the bottom region index. Defaults to None.
        text_prompt (str): The text prompt for the controlnet.
        modality_tensor (torch.Tensor): Int Tensor specifying the modality.
        num_inference_steps (int): Number of inference steps for the diffusion process.
        autoencoder_sliding_window_infer_size (list, optional): Size of the sliding window for inference. Defaults to [96, 96, 96].
        autoencoder_sliding_window_infer_overlap (float, optional): Overlap ratio for sliding window inference. Defaults to 0.6667.

    Returns:
        tuple: A tuple containing the synthetic image and its corresponding label.
    """
    # CT image intensity range
    a_min = -1000
    a_max = 1000
    # autoencoder output intensity range
    b_min = 0.0
    b_max = 1

    include_body_region = diffusion_unet.include_top_region_index_input
    include_modality = diffusion_unet.num_class_embeds is not None

    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        logging.info("---- Start generating latent features... ----")
        start_time = time.time()
        # generate segmentation mask
        combine_label = combine_label_or.to(device)
        if (
            output_size[0] != combine_label.shape[2]
            or output_size[1] != combine_label.shape[3]
            or output_size[2] != combine_label.shape[4]
        ):
            logging.info(
                "output_size is not a desired value. Need to interpolate the mask to match with output_size. The result image will be very low quality."
            )
            combine_label = torch.nn.functional.interpolate(combine_label, size=output_size, mode="nearest")

        controlnet_cond_vis = binarize_labels(combine_label.long()).half()

        # Generate random noise
        latents = initialize_noise_latents(latent_shape, device) * noise_factor

        # synthesize latents
        if isinstance(noise_scheduler, RFlowScheduler):
            noise_scheduler.set_timesteps(
                num_inference_steps=num_inference_steps,
                input_img_size_numel=torch.prod(torch.tensor(latents.shape[2:])),
            )
        else:
            noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        if isinstance(noise_scheduler, DDPMScheduler) and num_inference_steps < noise_scheduler.num_train_timesteps:
            warnings.warn(
                "**************************************************************\n"
                "* WARNING: Image noise_scheduler is a DDPMScheduler.\n"
                "* We expect num_inference_steps = noise_scheduler.num_train_timesteps"
                f" = {noise_scheduler.num_train_timesteps}.\n"
                f"* Yet got num_inference_steps = {num_inference_steps}.\n"
                "* The generated image quality is not guaranteed.\n"
                "**************************************************************"
            )

        all_timesteps = noise_scheduler.timesteps
        all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))
        progress_bar = tqdm(
            zip(all_timesteps, all_next_timesteps),
            total=min(len(all_timesteps), len(all_next_timesteps)),
        )
        for t, next_t in progress_bar:
            # get controlnet output
            # Create a dictionary to store the inputs
            controlnet_inputs = {
                "x": latents,
                "timesteps": torch.Tensor((t,)).to(device),
                "controlnet_cond": controlnet_cond_vis,
                "context": context,
            }
            if include_modality:
                controlnet_inputs.update(
                    {
                        "class_labels": modality_tensor,
                    }
                )
            down_block_res_samples, mid_block_res_sample = controlnet(**controlnet_inputs)

            # get diffusion network output
            # Create a dictionary to store the inputs
            unet_inputs = {
                "x": latents,
                "timesteps": torch.Tensor((t,)).to(device),
                "spacing_tensor": spacing_tensor,
                "down_block_additional_residuals": down_block_res_samples,
                "mid_block_additional_residual": mid_block_res_sample,
                "context": context,
            }
            # Add extra arguments if include_body_region is True
            if include_body_region:
                unet_inputs.update(
                    {
                        "top_region_index_tensor": top_region_index_tensor,
                        "bottom_region_index_tensor": bottom_region_index_tensor,
                    }
                )
            if include_modality:
                unet_inputs.update(
                    {
                        "class_labels": modality_tensor,
                    }
                )
            model_output = diffusion_unet(**unet_inputs)

            if not isinstance(noise_scheduler, RFlowScheduler):
                latents, _ = noise_scheduler.step(model_output, t, latents)  # type: ignore
            else:
                latents, _ = noise_scheduler.step(model_output, t, latents, next_t)  # type: ignore
        end_time = time.time()
        logging.info(f"---- DM/ControlNet Latent features generation time: {end_time - start_time} seconds ----")
        del (
            unet_inputs,
            controlnet_inputs,
            model_output,
            controlnet_cond_vis,
            down_block_res_samples,
            mid_block_res_sample,
        )
        gc.collect()
        torch.cuda.empty_cache()

        # decode latents to synthesized images
        logging.info("---- Start decoding latent features into images... ----")
        start_time = time.time()

        # ## print the latents shape
        # print(f"Latents shape before decoding: {latents.shape}")
        # print(type(latents))


        inferer = SlidingWindowInferer(
            roi_size=autoencoder_sliding_window_infer_size,
            sw_batch_size=1,
            progress=True,
            mode="gaussian",
            overlap=autoencoder_sliding_window_infer_overlap,
            sw_device=device,
            device=torch.device("cpu"),
        )
        synthetic_images = dynamic_infer(inferer, recon_model, latents)
        synthetic_images = torch.clip(synthetic_images, b_min, b_max).cpu()
        end_time = time.time()
        logging.info(f"---- Image VAE decoding time: {end_time - start_time} seconds ----")

        ## post processing:
        # project output to [0, 1]
        synthetic_images = (synthetic_images - b_min) / (b_max - b_min)
        # project output to [-1000, 1000]
        synthetic_images = synthetic_images * (a_max - a_min) + a_min

        print("---- Start piecewise linear mapping post-processing... ----")
        
        
        ## post-processing with piecewise linear mapping
        data = synthetic_images.squeeze().cpu().detach().numpy()
        data = revise_transform(data, method="piecewise_linear_mapping")
        synthetic_images = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(device)


        # regularize background intensities
        synthetic_images = crop_img_body_mask(synthetic_images, combine_label)
        torch.cuda.empty_cache()

    return synthetic_images, combine_label


def ldm_conditional_sample_one_embed(
    diffusion_unet,
    controlnet,
    noise_scheduler,
    scale_factor,
    device,
    combine_label_or,
    spacing_tensor,
    latent_shape,
    output_size,
    noise_factor,
    top_region_index_tensor=None,
    bottom_region_index_tensor=None,
    context=None,
    modality_tensor=None,
    num_inference_steps=1000,
):
    """
    Generate a single synthetic image using a latent diffusion model with controlnet.

    Args:
        diffusion_unet (nn.Module): The diffusion U-Net model.
        controlnet (nn.Module): The controlnet model.
        noise_scheduler: The noise scheduler for the diffusion process.
        scale_factor (float): Scaling factor for the latent space.
        device (torch.device): The device to run the computation on.
        combine_label_or (torch.Tensor): The combined label tensor.
        spacing_tensor (torch.Tensor): Tensor specifying the spacing.
        latent_shape (tuple): The shape of the latent space.
        output_size (tuple): The desired output size of the image.
        noise_factor (float): Factor to scale the initial noise.
        top_region_index_tensor (torch.Tensor): Tensor specifying the top region index. Defaults to None.
        bottom_region_index_tensor (torch.Tensor): Tensor specifying the bottom region index. Defaults to None.
        text_prompt (str): The text prompt for the controlnet.
        modality_tensor (torch.Tensor): Int Tensor specifying the modality.
        num_inference_steps (int): Number of inference steps for the diffusion process.

    Returns:
        tuple: A tuple containing the synthetic embeddings and its corresponding label.
    """
    # CT image intensity range
    a_min = -1000
    a_max = 1000
    # autoencoder output intensity range
    b_min = 0.0
    b_max = 1

    include_body_region = diffusion_unet.include_top_region_index_input
    include_modality = diffusion_unet.num_class_embeds is not None


    with torch.no_grad(), torch.amp.autocast("cuda"):
        logging.info("---- Start generating latent features... ----")
        start_time = time.time()
        # generate segmentation mask
        combine_label = combine_label_or.to(device)
        if (
            output_size[0] != combine_label.shape[2]
            or output_size[1] != combine_label.shape[3]
            or output_size[2] != combine_label.shape[4]
        ):
            logging.info(
                "output_size is not a desired value. Need to interpolate the mask to match with output_size. The result image will be very low quality."
            )
            combine_label = torch.nn.functional.interpolate(combine_label, size=output_size, mode="nearest")

        controlnet_cond_vis = binarize_labels(combine_label.long()).half()

        # Generate random noise
        latents = initialize_noise_latents(latent_shape, device) * noise_factor

        # synthesize latents
        if isinstance(noise_scheduler, RFlowScheduler):
            noise_scheduler.set_timesteps(
                num_inference_steps=num_inference_steps,
                input_img_size_numel=torch.prod(torch.tensor(latents.shape[2:])),
            )
        else:
            noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        if isinstance(noise_scheduler, DDPMScheduler) and num_inference_steps < noise_scheduler.num_train_timesteps:
            warnings.warn(
                "**************************************************************\n"
                "* WARNING: Image noise_scheduler is a DDPMScheduler.\n"
                "* We expect num_inference_steps = noise_scheduler.num_train_timesteps"
                f" = {noise_scheduler.num_train_timesteps}.\n"
                f"* Yet got num_inference_steps = {num_inference_steps}.\n"
                "* The generated image quality is not guaranteed.\n"
                "**************************************************************"
            )

        all_timesteps = noise_scheduler.timesteps
        all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))
        progress_bar = tqdm(
            zip(all_timesteps, all_next_timesteps),
            total=min(len(all_timesteps), len(all_next_timesteps)),
        )
        for t, next_t in progress_bar:
            # get controlnet output
            # Create a dictionary to store the inputs
            controlnet_inputs = {
                "x": latents,
                "timesteps": torch.Tensor((t,)).to(device),
                "controlnet_cond": controlnet_cond_vis,
                "context": context,
            }
            if include_modality:
                controlnet_inputs.update(
                    {
                        "class_labels": modality_tensor,
                    }
                )
            down_block_res_samples, mid_block_res_sample = controlnet(**controlnet_inputs)

            # get diffusion network output
            # Create a dictionary to store the inputs
            unet_inputs = {
                "x": latents,
                "timesteps": torch.Tensor((t,)).to(device),
                "spacing_tensor": spacing_tensor,
                "down_block_additional_residuals": down_block_res_samples,
                "mid_block_additional_residual": mid_block_res_sample,
                "context": context,
            }
            # Add extra arguments if include_body_region is True
            if include_body_region:
                unet_inputs.update(
                    {
                        "top_region_index_tensor": top_region_index_tensor,
                        "bottom_region_index_tensor": bottom_region_index_tensor,
                    }
                )
            if include_modality:
                unet_inputs.update(
                    {
                        "class_labels": modality_tensor,
                    }
                )
            model_output = diffusion_unet(**unet_inputs)

            if not isinstance(noise_scheduler, RFlowScheduler):
                latents, _ = noise_scheduler.step(model_output, t, latents)  # type: ignore
            else:
                latents, _ = noise_scheduler.step(model_output, t, latents, next_t)  # type: ignore
        end_time = time.time()
        logging.info(f"---- DM/ControlNet Latent features generation time: {end_time - start_time} seconds ----")
        del (
            unet_inputs,
            controlnet_inputs,
            model_output,
            controlnet_cond_vis,
            down_block_res_samples,
            mid_block_res_sample,
        )
        gc.collect()
        torch.cuda.empty_cache()


    return latents, combine_label



def ldm_conditional_sample_one_image(
    autoencoder,
    diffusion_unet,
    controlnet,
    noise_scheduler,
    scale_factor,
    device,
    combine_label_or,
    spacing_tensor,
    latent_shape,
    output_size,
    noise_factor,
    top_region_index_tensor=None,
    bottom_region_index_tensor=None,
    context=None,
    modality_tensor=None,
    num_inference_steps=1000,
    autoencoder_sliding_window_infer_size=[96, 96, 96],
    autoencoder_sliding_window_infer_overlap=0.6667,
):
    """
    Generate a single synthetic image using a latent diffusion model with controlnet.

    Args:
        autoencoder (nn.Module): The autoencoder model.
        diffusion_unet (nn.Module): The diffusion U-Net model.
        controlnet (nn.Module): The controlnet model.
        noise_scheduler: The noise scheduler for the diffusion process.
        scale_factor (float): Scaling factor for the latent space.
        device (torch.device): The device to run the computation on.
        combine_label_or (torch.Tensor): The combined label tensor.
        spacing_tensor (torch.Tensor): Tensor specifying the spacing.
        latent_shape (tuple): The shape of the latent space.
        output_size (tuple): The desired output size of the image.
        noise_factor (float): Factor to scale the initial noise.
        top_region_index_tensor (torch.Tensor): Tensor specifying the top region index. Defaults to None.
        bottom_region_index_tensor (torch.Tensor): Tensor specifying the bottom region index. Defaults to None.
        text_prompt (str): The text prompt for the controlnet.
        modality_tensor (torch.Tensor): Int Tensor specifying the modality.
        num_inference_steps (int): Number of inference steps for the diffusion process.
        autoencoder_sliding_window_infer_size (list, optional): Size of the sliding window for inference. Defaults to [96, 96, 96].
        autoencoder_sliding_window_infer_overlap (float, optional): Overlap ratio for sliding window inference. Defaults to 0.6667.

    Returns:
        tuple: A tuple containing the synthetic image and its corresponding label.
    """
    # CT image intensity range
    a_min = -1000
    a_max = 1000
    # autoencoder output intensity range
    b_min = 0.0
    b_max = 1

    include_body_region = diffusion_unet.include_top_region_index_input
    include_modality = diffusion_unet.num_class_embeds is not None

    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        logging.info("---- Start generating latent features... ----")
        start_time = time.time()
        # generate segmentation mask
        combine_label = combine_label_or.to(device)
        if (
            output_size[0] != combine_label.shape[2]
            or output_size[1] != combine_label.shape[3]
            or output_size[2] != combine_label.shape[4]
        ):
            logging.info(
                "output_size is not a desired value. Need to interpolate the mask to match with output_size. The result image will be very low quality."
            )
            combine_label = torch.nn.functional.interpolate(combine_label, size=output_size, mode="nearest")

        controlnet_cond_vis = binarize_labels(combine_label.long()).half()

        # Generate random noise
        latents = initialize_noise_latents(latent_shape, device) * noise_factor

        # synthesize latents
        if isinstance(noise_scheduler, RFlowScheduler):
            noise_scheduler.set_timesteps(
                num_inference_steps=num_inference_steps,
                input_img_size_numel=torch.prod(torch.tensor(latents.shape[2:])),
            )
        else:
            noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        if isinstance(noise_scheduler, DDPMScheduler) and num_inference_steps < noise_scheduler.num_train_timesteps:
            warnings.warn(
                "**************************************************************\n"
                "* WARNING: Image noise_scheduler is a DDPMScheduler.\n"
                "* We expect num_inference_steps = noise_scheduler.num_train_timesteps"
                f" = {noise_scheduler.num_train_timesteps}.\n"
                f"* Yet got num_inference_steps = {num_inference_steps}.\n"
                "* The generated image quality is not guaranteed.\n"
                "**************************************************************"
            )

        all_timesteps = noise_scheduler.timesteps
        all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))
        progress_bar = tqdm(
            zip(all_timesteps, all_next_timesteps),
            total=min(len(all_timesteps), len(all_next_timesteps)),
        )
        for t, next_t in progress_bar:
            # get controlnet output
            # Create a dictionary to store the inputs
            controlnet_inputs = {
                "x": latents,
                "timesteps": torch.Tensor((t,)).to(device),
                "controlnet_cond": controlnet_cond_vis,
                "context": context,
            }
            if include_modality:
                controlnet_inputs.update(
                    {
                        "class_labels": modality_tensor,
                    }
                )
            down_block_res_samples, mid_block_res_sample = controlnet(**controlnet_inputs)

            # get diffusion network output
            # Create a dictionary to store the inputs
            unet_inputs = {
                "x": latents,
                "timesteps": torch.Tensor((t,)).to(device),
                "spacing_tensor": spacing_tensor,
                "down_block_additional_residuals": down_block_res_samples,
                "mid_block_additional_residual": mid_block_res_sample,
                "context": context,
            }
            # Add extra arguments if include_body_region is True
            if include_body_region:
                unet_inputs.update(
                    {
                        "top_region_index_tensor": top_region_index_tensor,
                        "bottom_region_index_tensor": bottom_region_index_tensor,
                    }
                )
            if include_modality:
                unet_inputs.update(
                    {
                        "class_labels": modality_tensor,
                    }
                )
            model_output = diffusion_unet(**unet_inputs)

            if not isinstance(noise_scheduler, RFlowScheduler):
                latents, _ = noise_scheduler.step(model_output, t, latents)  # type: ignore
            else:
                latents, _ = noise_scheduler.step(model_output, t, latents, next_t)  # type: ignore
        end_time = time.time()
        logging.info(f"---- DM/ControlNet Latent features generation time: {end_time - start_time} seconds ----")
        del (
            unet_inputs,
            controlnet_inputs,
            model_output,
            controlnet_cond_vis,
            down_block_res_samples,
            mid_block_res_sample,
        )
        gc.collect()
        torch.cuda.empty_cache()

        # decode latents to synthesized images
        logging.info("---- Start decoding latent features into images... ----")
        start_time = time.time()

        inferer = SlidingWindowInferer(
            roi_size=autoencoder_sliding_window_infer_size,
            sw_batch_size=1,
            progress=True,
            mode="gaussian",
            overlap=autoencoder_sliding_window_infer_overlap,
            sw_device=device,
            device=torch.device("cpu"),
        )
        synthetic_images = dynamic_infer(inferer, recon_model, latents)
        synthetic_images = torch.clip(synthetic_images, b_min, b_max).cpu()
        end_time = time.time()
        logging.info(f"---- Image VAE decoding time: {end_time - start_time} seconds ----")

        ## post processing:
        # project output to [0, 1]
        synthetic_images = (synthetic_images - b_min) / (b_max - b_min)
        # project output to [-1000, 1000]
        synthetic_images = synthetic_images * (a_max - a_min) + a_min
        # regularize background intensities
        synthetic_images = crop_img_body_mask(synthetic_images, combine_label)
        torch.cuda.empty_cache()

    return synthetic_images, combine_label





