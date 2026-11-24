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

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime

import torch
import torch.distributed as dist
from monai.data import MetaTensor, decollate_batch
from monai.networks.utils import copy_model_state
from monai.transforms import SaveImage
from monai.utils import RankFilter
from transformers import AutoTokenizer, T5EncoderModel

from sample import check_input, ldm_conditional_sample_one_image
from utils import define_instance, prepare_maisi_controlnet_json_dataloader, setup_ddp
import torch.nn.functional as F

# PYTHONPATH=$PYTHONPATH:/path/to/maisi_3d_ct_synthesis/scripts python scripts/infer_controlnet.py

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="maisi.controlnet.infer")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./configs/environment_maisi_controlnet_inference_demo.json",
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
        default="./configs/config_maisi_controlnet_train.json",
        help="config json file that stores training hyper-parameters",
    )
    parser.add_argument("-g", "--gpus", default=2, type=int, help="number of gpus per node")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="skip samples whose image_id already has output files in output_dir",
    )

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

    # Build set of already-generated image_ids to skip before partitioning
    skip_ids = set()
    if args.skip_existing and os.path.isdir(args.output_dir):
        for f in os.listdir(args.output_dir):
            if f.endswith("_image.nii.gz"):
                # filename pattern: {image_id}_pred_{timestamp}_image.nii.gz
                image_id = f.rsplit("_pred_", 1)[0]
                skip_ids.add(image_id)
        print(f"[Rank {rank}] Found {len(skip_ids)} already-generated image_ids, will skip those.")

    # Step 1: set data loader
    val_loader = prepare_maisi_controlnet_json_dataloader(
        json_data_list=args.json_data_list,
        data_base_dir=args.data_base_dir,
        rank=rank,
        world_size=world_size,
        batch_size=args.controlnet_train["batch_size"],
        cache_rate=args.controlnet_train["cache_rate"],
        fold=args.controlnet_train["fold"],
        skip_ids=skip_ids if args.skip_existing else None,
    )

    # Step 2: define AE, diffusion model and controlnet
    # define AE
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    # load trained autoencoder model
    if args.trained_autoencoder_path is not None:
        if not os.path.exists(args.trained_autoencoder_path):
            raise ValueError("Please download the autoencoder checkpoint.")
        autoencoder_ckpt = torch.load(args.trained_autoencoder_path, weights_only=True)
        autoencoder.load_state_dict(autoencoder_ckpt)
        logger.info(f"Load trained diffusion model from {args.trained_autoencoder_path}.")
    else:
        logger.info("trained autoencoder model is not loaded.")

    # define diffusion Model
    unet = define_instance(args, "diffusion_unet_def").to(device)
    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None

    # load trained diffusion model
    if args.trained_diffusion_path is not None:
        if not os.path.exists(args.trained_diffusion_path):
            raise ValueError("Please download the trained diffusion unet checkpoint.")
        diffusion_model_ckpt = torch.load(args.trained_diffusion_path, map_location=device, weights_only=False)
        unet.load_state_dict(diffusion_model_ckpt["unet_state_dict"])
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
    tokenizer = AutoTokenizer.from_pretrained("luqh/ClinicalT5-base", model_max_length=512)
    text_model_path = os.environ.get("CF_CLINICALT5", "/path/to/inspect2/pe_training_data_all/models/clinicalt5-pytorch")
    text_encoder = T5EncoderModel.from_pretrained(text_model_path)

    text_encoder = text_encoder.to(device) 
    text_encoder.eval()   
    # --- PE SEGMENTATION SETUP ---
    os.environ["nnUNet_results"] = os.environ.get("CF_NNUNET_RESULTS", "/path/to/PulmonaryEmbolismSegmentation/dataset/nnUNet_results")
    os.environ["nnUNet_raw"] = os.environ.get("CF_NNUNET_INPUT", "/path/to/inspect2/CTPA_nnunet_input")
    os.environ["nnUNet_preprocessed"] = os.environ.get("CF_NNUNET_INPUT", "/path/to/inspect2/CTPA_nnunet_input")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # The detector is loaded when the first entry with pe_status = 1 is reached, so entries
    # with pe_status = 0 can be generated without the detector weights.
    pe_predictor = None

    def load_pe_predictor():
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_mirroring=False,
            verbose=False,
            perform_everything_on_device=True,
            device=device
        )
        predictor.initialize_from_trained_model_folder(
            os.path.join(
                os.environ["nnUNet_results"],
                "Dataset256_PE/nnUNetTrainer_500epochs__nnUNetResEncUNetPlans_80G__3d_fullres",
            ),
            use_folds="all",
            checkpoint_name="checkpoint_final.pth",
        )
        logger.info("Initialized nnUNet PE segmentation predictor.")
        return predictor

    # Step 3: inference
    autoencoder.eval()
    controlnet.eval()
    unet.eval()

    for batch in val_loader:
        # get label mask
        labels = batch["segmentation"].to(device)
        filename = batch["image_id"][0]
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
        labels_resized = F.interpolate(labels.as_tensor().to(torch.uint8), size=(output_size[0], output_size[1], output_size[2]), mode="nearest")            #if dim is 4 unsqueeze it

        latent_shape = (args.latent_channels, output_size[0] // 4, output_size[1] // 4, output_size[2] // 4)
        # check if output_size and out_spacing are valid.
        check_input(None, None, None, output_size, out_spacing, None)
        print(f"output_size: {output_size}")
        # --- TEXT CONDITIONING ---
        prompts = batch["text_prompt"]
        tokens = tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            encoder_outputs = text_encoder(**tokens)
            text_emb = encoder_outputs.last_hidden_state  # [B, seq_len, 768]

        # generate a single synthetic image using a latent diffusion model with controlnet.
        synthetic_images, _ = ldm_conditional_sample_one_image(
            autoencoder=autoencoder,
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
            autoencoder_sliding_window_infer_size=args.controlnet_infer["autoencoder_sliding_window_infer_size"],
            autoencoder_sliding_window_infer_overlap=args.controlnet_infer["autoencoder_sliding_window_infer_overlap"]
        )
        # save image/label pairs
        labels = decollate_batch(batch)[0]["segmentation"]
        output_postfix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        labels.meta["filename_or_obj"] = f"{filename}_pred.nii.gz"
        # The label map was resized to output_size, so the generated grid has the label's
        # affine with axis i scaled by (label size / output size); the origin is unchanged.
        out_meta = dict(labels.meta)
        out_affine = labels.affine.clone()
        for i in range(3):
            out_affine[:3, i] *= labels.shape[i + 1] / output_size[i]
        out_meta["affine"] = out_affine
        synthetic_images = MetaTensor(synthetic_images.squeeze(0), meta=out_meta)
        img_saver = SaveImage(
            output_dir=args.output_dir,
            output_postfix=output_postfix + "_image",
            separate_folder=False,
        )
        img_saver(synthetic_images)
        # save the resized label map (the generator input) on the same grid as the image
        label_saver = SaveImage(
            output_dir=args.output_dir,
            output_postfix=output_postfix + "_label",
            separate_folder=False,
        )
        label_saver(MetaTensor(labels_resized[0].cpu(), meta=dict(out_meta)))

        # Step 4: run PE segmentation on synthetic image if PE-positive
        pe_status = batch["pe_status"][0].item() if torch.is_tensor(batch["pe_status"][0]) else batch["pe_status"][0]
        if pe_status == 1:
            if pe_predictor is None:
                pe_predictor = load_pe_predictor()
            saved_image_name = f"{filename}_pred_{output_postfix}_image.nii.gz"
            saved_image_path = os.path.join(args.output_dir, saved_image_name)

            tmp_dir = tempfile.mkdtemp(dir=args.output_dir)
            tmp_in = os.path.join(tmp_dir, "input")
            tmp_out = os.path.join(tmp_dir, "output")
            os.makedirs(tmp_in)
            os.makedirs(tmp_out)
            shutil.copy(saved_image_path, os.path.join(tmp_in, f"{filename}_0000.nii.gz"))

            # nnUNet spawns subprocesses that call os.getcwd(); ensure cwd is valid
            orig_cwd = os.getcwd()
            os.chdir(args.output_dir)
            pe_predictor.predict_from_files(
                list_of_lists_or_source_folder=tmp_in,
                output_folder_or_list_of_truncated_output_files=tmp_out,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
            )
            os.chdir(orig_cwd)

            pe_seg_src = os.path.join(tmp_out, f"{filename}.nii.gz")
            pe_seg_dst = os.path.join(args.output_dir, f"{filename}_pred_{output_postfix}_pe_seg.nii.gz")
            shutil.move(pe_seg_src, pe_seg_dst)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"Saved PE segmentation: {pe_seg_dst}")

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
