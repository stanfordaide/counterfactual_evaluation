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
import time
import warnings
import gc

import torch
from monai.inferers.inferer import DiffusionInferer
from tqdm import tqdm
from monai.inferers.inferer import SlidingWindowInferer
from monai.networks.schedulers import RFlowScheduler, DDPMScheduler

from utils import (
    binarize_labels,
    general_mask_generation_post_process,
    remap_labels,
    dynamic_infer,
)


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


def ldm_conditional_sample_one_mask(
    autoencoder,
    diffusion_unet,
    noise_scheduler,
    scale_factor,
    anatomy_size,
    device,
    latent_shape,
    label_dict_remap_json,
    num_inference_steps=1000,
    autoencoder_sliding_window_infer_size=[96, 96, 96],
    autoencoder_sliding_window_infer_overlap=0.6667,
):
    """
    Generate a single synthetic mask using a latent diffusion model.

    Args:
        autoencoder (nn.Module): The autoencoder model.
        diffusion_unet (nn.Module): The diffusion U-Net model.
        noise_scheduler: The noise scheduler for the diffusion process.
        scale_factor (float): Scaling factor for the latent space.
        anatomy_size (torch.Tensor): Tensor specifying the desired anatomy sizes.
        device (torch.device): The device to run the computation on.
        latent_shape (tuple): The shape of the latent space.
        label_dict_remap_json (str): Path to the JSON file for label remapping.
        num_inference_steps (int): Number of inference steps for the diffusion process.
        autoencoder_sliding_window_infer_size (list, optional): Size of the sliding window for inference. Defaults to [96, 96, 96].
        autoencoder_sliding_window_infer_overlap (float, optional): Overlap ratio for sliding window inference. Defaults to 0.6667.

    Returns:
        torch.Tensor: The generated synthetic mask.
    """
    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        # Generate random noise
        latents = initialize_noise_latents(latent_shape, device)
        anatomy_size = torch.FloatTensor(anatomy_size).unsqueeze(0).unsqueeze(0).half().to(device)
        # synthesize latents
        if isinstance(noise_scheduler, DDPMScheduler) and num_inference_steps < noise_scheduler.num_train_timesteps:
            warnings.warn(
                "**************************************************************\n"
                "* WARNING: Mask noise_scheduler is a DDPMScheduler.\n"
                "* We expect num_inference_steps = noise_scheduler.num_train_timesteps"
                f" = {noise_scheduler.num_train_timesteps}.\n"
                f"* Yet got num_inference_steps = {num_inference_steps}.\n"
                "* The generated image quality is not guaranteed.\n"
                "**************************************************************"
            )

        noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        # mask generator is DDPM
        inferer_ddpm = DiffusionInferer(noise_scheduler)
        latents = inferer_ddpm.sample(
            input_noise=latents,
            diffusion_model=diffusion_unet,
            scheduler=noise_scheduler,
            verbose=True,
            conditioning=anatomy_size.to(device),
        )

        inferer = SlidingWindowInferer(
            roi_size=autoencoder_sliding_window_infer_size,
            sw_batch_size=1,
            progress=True,
            mode="gaussian",
            overlap=autoencoder_sliding_window_infer_overlap,
            sw_device=device,
            device=torch.device("cpu"),
        )
        synthetic_mask = dynamic_infer(inferer, recon_model, latents)
        synthetic_mask = torch.softmax(synthetic_mask, dim=1)
        synthetic_mask = torch.argmax(synthetic_mask, dim=1, keepdim=True)
        # mapping raw index to 132 labels
        synthetic_mask = remap_labels(synthetic_mask, label_dict_remap_json)

        ###### post process #####
        data = synthetic_mask.squeeze().cpu().detach().numpy()

        labels = [23, 24, 26, 27, 128]
        target_tumor_label = None
        for index, size in enumerate(anatomy_size[0, 0, 5:10]):
            if size.item() != -1.0:
                target_tumor_label = labels[index]

        logging.info(f"target_tumor_label for postprocess:{target_tumor_label}")
        data = general_mask_generation_post_process(data, target_tumor_label=target_tumor_label, device=device)
        synthetic_mask = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(device)

    return synthetic_mask


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


def filter_mask_with_organs(combine_label, anatomy_list):
    """
    Filter a mask to only include specified organs.

    Args:
        combine_label (torch.Tensor): The input mask.
        anatomy_list (list): List of organ labels to keep.

    Returns:
        torch.Tensor: The filtered mask.
    """
    # final output mask file has shape of output_size, contains labels in anatomy_list
    # it is already interpolated to target size
    combine_label = combine_label.long()
    # filter out the organs that are not in anatomy_list
    for i in range(len(anatomy_list)):
        organ = anatomy_list[i]
        # replace it with a negative value so it will get mixed
        combine_label[combine_label == organ] = -(i + 1)
    # zero-out voxels with value not in anatomy_list
    combine_label[combine_label > 0] = 0
    # output positive values
    combine_label = -combine_label
    return combine_label


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


def check_input(
    body_region,
    anatomy_list,
    label_dict_json,
    output_size,
    spacing,
    controllable_anatomy_size=[("pancreas", 0.5)],
):
    """
    Validate input parameters for image generation.

    Args:
        body_region (list): List of body regions.
        anatomy_list (list): List of anatomical structures.
        label_dict_json (str): Path to the label dictionary JSON file.
        output_size (tuple): Desired output size of the image.
        spacing (tuple): Desired voxel spacing.
        controllable_anatomy_size (list): List of tuples specifying controllable anatomy sizes.

    Raises:
        ValueError: If any input parameter is invalid.
    """
    # check output_size and spacing format
    if output_size[0] != output_size[1]:
        raise ValueError(f"The first two components of output_size need to be equal, yet got {output_size}.")
    if (output_size[0] not in [256, 384, 512]) or (output_size[2] not in [128, 256, 384, 512, 640, 768]):
        raise ValueError(
            f"The output_size[0] have to be chosen from [256, 384, 512], and output_size[2] have to be chosen from [128, 256, 384, 512, 640, 768], yet got {output_size}."
        )

    if spacing[0] != spacing[1]:
        raise ValueError(f"The first two components of spacing need to be equal, yet got {spacing}.")
    if spacing[0] < 0.5 or spacing[0] > 3.0 or spacing[2] < 0.5 or spacing[2] > 5.0:
        raise ValueError(
            f"spacing[0] have to be between 0.5 and 3.0 mm, spacing[2] have to be between 0.5 and 5.0 mm, yet got {spacing}."
        )

    if output_size[0] * spacing[0] < 256:
        FOV = [output_size[axis] * spacing[axis] for axis in range(3)]
        raise ValueError(
            f"`'spacing'({spacing}mm) and 'output_size'({output_size}) together decide the output field of view (FOV). The FOV will be {FOV}mm. We recommend the FOV in x and y axis to be at least 256mm for head, and at least 384mm for other body regions like abdomen. There is no such restriction for z-axis."
        )

    if controllable_anatomy_size == None:
        logging.info(f"`controllable_anatomy_size` is not provided.")
        return

    # check controllable_anatomy_size format
    if len(controllable_anatomy_size) > 10:
        raise ValueError(
            f"The length of list controllable_anatomy_size has to be less than 10. Yet got length equal to {len(controllable_anatomy_size)}."
        )
    available_controllable_organ = [
        "liver",
        "gallbladder",
        "stomach",
        "pancreas",
        "colon",
    ]
    available_controllable_tumor = [
        "hepatic tumor",
        "bone lesion",
        "lung tumor",
        "colon cancer primaries",
        "pancreatic tumor",
    ]
    available_controllable_anatomy = available_controllable_organ + available_controllable_tumor
    controllable_tumor = []
    controllable_organ = []
    for controllable_anatomy_size_pair in controllable_anatomy_size:
        if controllable_anatomy_size_pair[0] not in available_controllable_anatomy:
            raise ValueError(
                f"The controllable_anatomy have to be chosen from {available_controllable_anatomy}, yet got {controllable_anatomy_size_pair[0]}."
            )
        if controllable_anatomy_size_pair[0] in available_controllable_tumor:
            controllable_tumor += [controllable_anatomy_size_pair[0]]
        if controllable_anatomy_size_pair[0] in available_controllable_organ:
            controllable_organ += [controllable_anatomy_size_pair[0]]
        if controllable_anatomy_size_pair[1] == -1:
            continue
        if controllable_anatomy_size_pair[1] < 0 or controllable_anatomy_size_pair[1] > 1.0:
            raise ValueError(
                f"The controllable size scale have to be between 0 and 1,0, or equal to -1, yet got {controllable_anatomy_size_pair[1]}."
            )
    if len(controllable_tumor + controllable_organ) != len(list(set(controllable_tumor + controllable_organ))):
        raise ValueError(f"Please do not repeat controllable_anatomy. Got {controllable_tumor + controllable_organ}.")
    if len(controllable_tumor) > 1:
        raise ValueError(f"Only one controllable tumor is supported. Yet got {controllable_tumor}.")

    if len(controllable_anatomy_size) > 0:
        logging.info(
            f"`controllable_anatomy_size` is not empty.\nWe will ignore `body_region` and `anatomy_list` and synthesize based on `controllable_anatomy_size`: ({controllable_anatomy_size})."
        )
    else:
        logging.info(
            f"`controllable_anatomy_size` is empty.\nWe will synthesize based on `body_region`: ({body_region}) and `anatomy_list`: ({anatomy_list})."
        )
        # check body_region format
        available_body_region = [
            "head",
            "chest",
            "thorax",
            "abdomen",
            "pelvis",
            "lower",
        ]
        for region in body_region:
            if region not in available_body_region:
                raise ValueError(
                    f"The components in body_region have to be chosen from {available_body_region}, yet got {region}."
                )

        # check anatomy_list format
        with open(label_dict_json) as f:
            label_dict = json.load(f)
        for anatomy in anatomy_list:
            if anatomy not in label_dict.keys():
                raise ValueError(
                    f"The components in anatomy_list have to be chosen from {label_dict.keys()}, yet got {anatomy}."
                )
    logging.info(f"The generate results will have voxel size to be {spacing}mm, volume size to be {output_size}.")

    return
