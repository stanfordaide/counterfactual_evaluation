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
import os
import argparse
import json
import logging
import os
import sys
from datetime import datetime

import torch
import torch.distributed as dist
from monai.data import MetaTensor, decollate_batch
from monai.networks.utils import copy_model_state
from monai.transforms import SaveImage
from monai.utils import RankFilter
from transformers import AutoTokenizer, T5EncoderModel
from transformers import T5Tokenizer


from sample import ReconModel, ldm_conditional_sample_one_embed, ldm_conditional_sample_one_image_with_piecewise_linear_postprocessing
from utils import define_instance, prepare_maisi_controlnet_json_dataloader, setup_ddp, prepare_maisi_controlnet_json_dataloader_for_inference
import torch.nn.functional as F
import numpy as np
import gc
import nibabel as nib
import time
from monai.inferers.inferer import SlidingWindowInferer
from utils import (
    binarize_labels,
    general_mask_generation_post_process,
    get_body_region_index_from_mask,
    remap_labels,
    dynamic_infer,
)





def revise_transform(data, method: str) -> np.ndarray:
    
    if method == "tanh_brain_wrapping":
        arr = data
        
        hu_min = -1000
        hu_max = 1000
        alpha = 1 / 200.0  ## brain windowing (W:80, L:40)
        center = 40.0
        
        arr = arr.astype(np.float32)
        arr = np.clip(arr / 1000.0, -0.999999, 0.999999)  # avoid inf values

        inv_tanh_arr = np.arctanh(arr) / alpha + center
        inv_tanh_arr = np.clip(inv_tanh_arr, hu_min, hu_max)
        out_arr = inv_tanh_arr.astype(np.int16)
        data = out_arr
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


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="maisi.controlnet.infer")
   
    
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./configs/environment_maisi_controlnet_infer_with_text_ICH_seg_piecewise_linear_mapping_CenterWeightedMSE.json",
        help="environment json file that stores environment path",
    )
    

    parser.add_argument(
        "-c",
        "--config-file",
        default="./configs/config_maisi3d-rflow.json",
        help="config json file that stores network hyper-parameters",
    )
    parser.add_argument(
        "-t",
        "--training-config",
        default="./configs/config_maisi_controlnet_train_headct_weighted_label.json",
        help="config json file that stores training hyper-parameters",
    )
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    

    

    args = parser.parse_args()

    # Step 0: configuration
    logger = logging.getLogger("maisi.controlnet.infer")
    # whether to use distributed data parallel
    use_ddp = args.gpus > 1
    if use_ddp:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = setup_ddp(rank, world_size)
        logger.addFilter(RankFilter())
    else:
        rank = 0
        world_size = 1
        device = torch.device(f"cuda:{rank}")

    torch.cuda.set_device(device)
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    logger.info(f"World_size: {world_size}")

    with open(args.environment_file, "r") as env_file:
        env_dict = json.load(env_file)
    with open(args.config_file, "r") as config_file:
        config_dict = json.load(config_file)
    with open(args.training_config, "r") as training_config_file:
        training_config_dict = json.load(training_config_file)

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)
    for k, v in training_config_dict.items():
        setattr(args, k, v)

    # Step 1: set data loader
    val_loader = prepare_maisi_controlnet_json_dataloader(
        json_data_list=args.json_data_list,
        data_base_dir=args.data_base_dir,
        rank=rank,
        world_size=world_size,
        batch_size=args.controlnet_train["batch_size"],
        cache_rate=args.controlnet_train["cache_rate"],
        fold=args.controlnet_train["fold"],
    )

   
    # Step 2: define AE, diffusion model and controlnet
    # define AE
    ## skip this step and define AE later for decoding

    # autoencoder = define_instance(args, "autoencoder_def").to(device)
    # # load trained autoencoder model
    # if args.trained_autoencoder_path is not None:
    #     if not os.path.exists(args.trained_autoencoder_path):
    #         raise ValueError("Please download the autoencoder checkpoint.")
    #     autoencoder_ckpt = torch.load(args.trained_autoencoder_path, weights_only=True)
    #     autoencoder.load_state_dict(autoencoder_ckpt)
    #     logger.info(f"Load trained autoencoder model from {args.trained_autoencoder_path}.")
    # else:
    #     logger.info("trained autoencoder model is not loaded.")

    #
    #  define diffusion Model
    unet = define_instance(args, "diffusion_unet_def").to(device)
    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None

    # load trained diffusion model
    if args.trained_diffusion_path is not None:
        if not os.path.exists(args.trained_diffusion_path):
            raise ValueError("Please download the trained diffusion unet checkpoint.")
        diffusion_model_ckpt = torch.load(args.trained_diffusion_path, map_location=device, weights_only=False)
        
        # unet.load_state_dict(diffusion_model_ckpt["unet_state_dict"]) previously use this one, I change to strict=False to avoid missing keys error becuase prev
        unet.load_state_dict(diffusion_model_ckpt["unet_state_dict"], strict=False)
        # load scale factor from diffusion model checkpoint
        scale_factor = diffusion_model_ckpt["scale_factor"]
        logger.info(f"Load trained diffusion model from {args.trained_diffusion_path}.")
        logger.info(f"loaded scale_factor from diffusion model ckpt -> {scale_factor}.")
    else:
        logger.info("trained diffusion model is not loaded.")
        scale_factor = 1.0
        logger.info(f"set scale_factor -> {scale_factor}.")

    # define ControlNet
    controlnet = define_instance(args, "controlnet_def").to(device)
    # copy weights from the DM to the controlnet
    copy_model_state(controlnet, unet.state_dict())
    # load trained controlnet model if it is provided
    if args.trained_controlnet_path is not None:
        if not os.path.exists(args.trained_controlnet_path):
            raise ValueError("Please download the trained ControlNet checkpoint.")
        controlnet.load_state_dict(
            torch.load(args.trained_controlnet_path, map_location=device, weights_only=False)["controlnet_state_dict"]
        )
        logger.info(f"load trained controlnet model from {args.trained_controlnet_path}")
    else:
        logger.info("trained controlnet is not loaded.")

    noise_scheduler = define_instance(args, "noise_scheduler")

    # --- TEXT CONDITIONING SETUP ---
    ## tokenizer = AutoTokenizer.from_pretrained("luqh/ClinicalT5-base", model_max_length=512)
    tokenizer_path = os.environ.get("CF_CLINICALT5_TOKENIZER", "/path/to/ClinicalT5-base/tokenizer")
    tokenizer = T5Tokenizer.from_pretrained(str(tokenizer_path), model_max_length=512)
    # tokenizer = AutoTokenizer.from_pretrained("luqh/ClinicalT5-base", model_max_length=512, trust_remote_code=True)
    ## does it means that the text encoder is finetuned during training? or we always keep it frozen?
    text_model_path = os.environ.get("CF_CLINICALT5", "/path/to/ClinicalT5-base/model")
    text_encoder = T5EncoderModel.from_pretrained(text_model_path)


    text_encoder = text_encoder.to(device)    
    # Step 3: inference
    # autoencoder.eval()
    controlnet.eval()
    unet.eval()



    os.makedirs(args.output_dir, exist_ok=True)

    ## create seperate dir inside output dir for generated embeddings and the decoded images. Because I want to seperate the geneation process and decoding process due to OOD issues.
    ##args.output_dir_embeddings = os.path.join(args.output_dir, "embeddings"+f"_{args.start_idx}_{args.end_idx}")
    args.output_dir_embeddings = os.path.join(args.output_dir, "embeddings")
    os.makedirs(args.output_dir_embeddings, exist_ok=True)

    

    for batch in val_loader:
        # get label mask
        labels = batch["segmentation"].to(device)
        # filename = batch["image_id"][0]
        filename = batch["image"][0]
        print(f"filename: {filename}")
        # get corresponding conditions
        if include_body_region:
            top_region_index_tensor = batch["top_region_index"].to(device)
            bottom_region_index_tensor = batch["bottom_region_index"].to(device)
        else:
            top_region_index_tensor = None
            bottom_region_index_tensor = None
        spacing_tensor = batch["spacing"].to(device)
        
        modality_tensor = args.controlnet_infer["modality"] * torch.ones((len(labels),), dtype=torch.long).to(device)
        out_spacing = tuple((batch["spacing"].squeeze().numpy() / 100).tolist())
        # get target dimension
        dim = batch["dim"]
        output_size = (dim[0].item(), dim[1].item(), dim[2].item())


        ## here I change the dim to samller size for resolving the memory issue
        ## this is learned from diff_model_create_training_data.py
        ## need to check again.
        # if output_size[2] == 512:
        #     output_size = (output_size[0], output_size[1], 384)

        labels_resized = F.interpolate(labels.as_tensor().to(torch.uint8), size=(output_size[0], output_size[1], output_size[2]), mode="nearest")            #if dim is 4 unsqueeze it

        latent_shape = (args.latent_channels, output_size[0] // 4, output_size[1] // 4, output_size[2] // 4)
        # check if output_size and out_spacing are valid.
        # check_input(None, None, None, output_size, out_spacing, None) ## I comment this out becasue headct output size does not fit the predefined sizes.
        print(f"output_size: {output_size}")
        # --- TEXT CONDITIONING ---
        prompts = batch["text_prompt"]
        tokens = tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            encoder_outputs = text_encoder(**tokens)
            text_emb = encoder_outputs.last_hidden_state  # [B, seq_len, 768]


        # # generate a single synthetic image using a latent diffusion model with controlnet.
        # synthetic_images, _ = ldm_conditional_sample_one_image_with_piecewise_linear_postprocessing(
        #     autoencoder=autoencoder,
        #     diffusion_unet=unet,
        #     controlnet=controlnet,
        #     noise_scheduler=noise_scheduler,
        #     scale_factor=scale_factor,
        #     device=device,
        #     combine_label_or=labels_resized,
        #     top_region_index_tensor=top_region_index_tensor,
        #     bottom_region_index_tensor=bottom_region_index_tensor,
        #     context=text_emb,
        #     spacing_tensor=spacing_tensor,
        #     modality_tensor=modality_tensor,
        #     latent_shape=latent_shape,
        #     output_size=output_size,
        #     noise_factor=1.0,
        #     num_inference_steps=args.controlnet_infer["num_inference_steps"],
        #     autoencoder_sliding_window_infer_size=args.controlnet_infer["autoencoder_sliding_window_infer_size"],
        #     autoencoder_sliding_window_infer_overlap=args.controlnet_infer["autoencoder_sliding_window_infer_overlap"]
        # )
        # # save image/label pairs

        ### since the above process requires too much memory, I directly generate the latent embeddings and save them for future decoding.
        synthetic_embeds, _ = ldm_conditional_sample_one_embed(
            diffusion_unet=unet,
            controlnet=controlnet,
            noise_scheduler=noise_scheduler,
            scale_factor=scale_factor,
            device=device,
            combine_label_or=labels_resized,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            context=text_emb,
            spacing_tensor=spacing_tensor,
            modality_tensor=modality_tensor,
            latent_shape=latent_shape,
            output_size=output_size,
            noise_factor=1.0,
            num_inference_steps=args.controlnet_infer["num_inference_steps"],
        )
        # save image/label pairs


        labels = decollate_batch(batch)[0]["segmentation"]
        output_postfix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        labels.meta["filename_or_obj"] = f"{filename}_pred.nii.gz"
        # The label map was resized to output_size, so the generated grid has the label's
        # affine with axis i scaled by (label size / output size); the origin is unchanged.
        # The embedding keeps this affine, and the decoding step below copies it to the CT.
        out_meta = dict(labels.meta)
        out_affine = labels.affine.clone()
        for i in range(3):
            out_affine[:3, i] *= labels.shape[i + 1] / output_size[i]
        out_meta["affine"] = out_affine
        synthetic_embeds = MetaTensor(synthetic_embeds.squeeze(0), meta=out_meta)

        img_saver = SaveImage(
            output_dir=args.output_dir_embeddings,
            output_postfix=output_postfix + "_image_embed",
            separate_folder=False,
        )
        img_saver(synthetic_embeds)
        # save the resized label map (the generator input) on the generated grid
        label_saver = SaveImage(
            output_dir=args.output_dir_embeddings,
            output_postfix=output_postfix + "_label",
            separate_folder=False,
        )
        label_saver(MetaTensor(labels_resized[0].cpu(), meta=dict(out_meta)))



    ## remove the tokenizer, text_encoder, unet and controlnet from GPU to free up space for decoding
    del unet
    del controlnet
    del tokenizer
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    ### now we want to decode the saved embeddings into images.
    print("Decoding saved embeddings into images...")
    ## create a directory to save decoded images
    args.output_dir_decoded_images = os.path.join(args.output_dir, "decoded_images")
    os.makedirs(args.output_dir_decoded_images, exist_ok=True)

    ## collect all saved embedding files, which end with _image_embed.nii.gz
    embed_files = [f for f in os.listdir(args.output_dir_embeddings) if f.endswith("_image_embed.nii.gz")]
    print(f"Total embedding files to decode: {len(embed_files)}")

    ## define AE
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    # load trained autoencoder model
    if args.trained_autoencoder_path is not None:
        if not os.path.exists(args.trained_autoencoder_path):
            raise ValueError("Please download the autoencoder checkpoint.")
        autoencoder_ckpt = torch.load(args.trained_autoencoder_path, weights_only=True)
        autoencoder.load_state_dict(autoencoder_ckpt)
        logger.info(f"Load trained autoencoder model from {args.trained_autoencoder_path}.")
    else:
        logger.info("trained autoencoder model is not loaded.")

    autoencoder_sliding_window_infer_size = args.controlnet_infer["autoencoder_sliding_window_infer_size"]
    autoencoder_sliding_window_infer_overlap = args.controlnet_infer["autoencoder_sliding_window_infer_overlap"]
    
    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)

    # CT image intensity range
    a_min = -1000
    a_max = 1000
    # autoencoder output intensity range
    b_min = 0.0
    b_max = 1

    for embed_file in embed_files:
        embed_path = os.path.join(args.output_dir_embeddings, embed_file)
        print(f"Decoding embedding file: {embed_path}")
        # load embedding
        embed_nii = nib.load(embed_path)
        embed_array = embed_nii.get_fdata()  # keep as nibabel object for later saving

        embed_tensor = torch.from_numpy(embed_array).float().to(device) ## (d,h,w,4)

        # print(f"embed_tensor shape before unsqueeze and permute: {embed_tensor.shape}")

        embed_tensor = embed_tensor.unsqueeze(0)  # add batch dimension  
        embed_tensor = embed_tensor.permute(0, 4, 1, 2, 3).contiguous()  ### (1,4,d,h,w)
        
        
        with torch.no_grad(), torch.amp.autocast("cuda"):
        
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
            synthetic_images = dynamic_infer(inferer, recon_model, embed_tensor)
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


        # # regularize background intensities
        # synthetic_images = crop_img_body_mask(synthetic_images, combine_label)  ## I did not do this step.
        torch.cuda.empty_cache()
        
        ## save decoded image
        output_file_name = embed_file.replace("_image_embed.nii.gz", f"_image_embed_decoded.nii.gz")
        output_path = os.path.join(args.output_dir_decoded_images, output_file_name)

        x = synthetic_images.squeeze().cpu().detach().numpy()
        x = x.astype(np.int16)
        new_nii = nib.Nifti1Image(x, affine=embed_nii.affine, header=embed_nii.header)
        nib.save(new_nii, output_path)
        
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
