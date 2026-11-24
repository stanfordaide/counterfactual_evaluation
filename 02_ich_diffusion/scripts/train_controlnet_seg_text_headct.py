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
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import monai
from monai.networks.utils import copy_model_state
from monai.utils import RankFilter
from monai.networks.schedulers import RFlowScheduler
from monai.networks.schedulers.ddpm import DDPMPredictionType
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer
from transformers import T5ForConditionalGeneration
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer, T5EncoderModel
import wandb
from utils import binarize_labels, define_instance, prepare_maisi_controlnet_json_dataloader, setup_ddp
from diff_model_setting import initialize_distributed, load_config, setup_logging
from transformers import T5Tokenizer


def main():
    
    # export PYTHONPATH=$(pwd):$PYTHONPATH

    # Enable anomaly detection to find gradient errors
    torch.autograd.set_detect_anomaly(True)
    
    parser = argparse.ArgumentParser(description="maisi.controlnet.training")

    parser.add_argument(
        "-e",
        "--environment-file",
        default="./configs/environment_maisi_controlnet_train_with_text_ICH_seg_piecewise_linear_mapping_CenterWeightedMSE.json",
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
    logger = logging.getLogger("maisi.controlnet.training")
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
    logger.info(f"Environment file: {args.environment_file}")
    logger.info(f"Config file: {args.config_file}")
    logger.info(f"Training config file: {args.training_config}")

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

    # After loading config_dict and training_config_dict
    # Ensure controlnet_def is present and merged/overridden as needed
    controlnet_def = config_dict.get("controlnet_def", {}).copy()
    controlnet_def.update(training_config_dict.get("controlnet_def", {}))
    setattr(args, "controlnet_def", controlnet_def)

    logger.info(f"Training config: {args.controlnet_train['weighted_loss']}")


    config = {
        'env_config_path': args.environment_file,
        'model_config_path': args.config_file,
        'model_def_path': args.training_config,
        'num_gpus': args.gpus,
        'amp': True,
        'batch_size': args.controlnet_train["batch_size"],
        'n_epochs': args.controlnet_train["n_epochs"],
        'lr': args.controlnet_train["lr"],
        'weighted_loss':args.controlnet_train["weighted_loss"],
    }
    exp_name = "controlnet_text_all_weighted_loss:" + str(args.controlnet_train["weighted_loss"])
    logger.info(f"Experiment name: {exp_name}")

    # initialize tensorboard writer
    if rank == 0:
        wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb.init(project="controlnet_training", 
                name=exp_name,
                config=config)

        tensorboard_path = os.path.join(args.tfevent_path, args.exp_name)
        Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_path)
        
    
    # Step 1: set data loader
    train_loader = prepare_maisi_controlnet_json_dataloader(
        json_data_list=args.json_data_list,
        data_base_dir=args.data_base_dir,
        rank=rank,
        world_size=world_size,
        batch_size=args.controlnet_train["batch_size"],
        cache_rate=args.controlnet_train["cache_rate"]
    )

    # Step 2: define diffusion model and controlnet
    # define diffusion Model
    unet = define_instance(args, "diffusion_unet_def").to(device)
    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None

    # load trained diffusion model
    if args.trained_diffusion_path is not None:
        if not os.path.exists(args.trained_diffusion_path):
            raise ValueError("Please download the trained diffusion unet checkpoint.")
        diffusion_model_ckpt = torch.load(args.trained_diffusion_path, map_location=device, weights_only=False)
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
            torch.load(args.trained_controlnet_path, map_location=device, weights_only=False)["controlnet_state_dict"],
            strict=False
        )
        logger.info(f"load trained controlnet model from {args.trained_controlnet_path}")
    else:
        logger.info("train controlnet model from scratch.")
    # we freeze the parameters of the diffusion model.
    for p in unet.parameters():
        p.requires_grad = False

    noise_scheduler = define_instance(args, "noise_scheduler")

    # --- TEXT CONDITIONING SETUP ---
    # Initialize CLIP tokenizer and encoder 
    # Set tokenizers parallelism to false to avoid warnings in multi-process environment
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # text_encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch16").to(device)
    # tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    # tokenizer = AutoTokenizer.from_pretrained("luqh/ClinicalT5-base", model_max_length=512)
    tokenizer_path = os.environ.get("CF_CLINICALT5_TOKENIZER", "/path/to/ClinicalT5-base/tokenizer")
    tokenizer = T5Tokenizer.from_pretrained(str(tokenizer_path), model_max_length=512)
    # text_encoder = T5EncoderModel.from_pretrained("luqh/ClinicalT5-base", from_flax=True).to(device)
    text_model_path = os.environ.get("CF_CLINICALT5", "/path/to/ClinicalT5-base/model")
    text_encoder = T5EncoderModel.from_pretrained(text_model_path)

    text_encoder = text_encoder.to(device)
    if use_ddp:
        controlnet = DDP(controlnet, device_ids=[device], output_device=rank, find_unused_parameters=False)

    # Step 3: training config
    weighted_loss = args.controlnet_train["weighted_loss"]
    if weighted_loss > 1.0:
        weighted_loss_label = args.controlnet_train["weighted_loss_label"]
    
    optimizer = torch.optim.AdamW(params=controlnet.parameters(), lr=args.controlnet_train["lr"])
    total_steps = (args.controlnet_train["n_epochs"] * len(train_loader.dataset)) / args.controlnet_train["batch_size"]
    logger.info(f"total number of training steps: {total_steps}.")

    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)

    # Step 4: training
    n_epochs = args.controlnet_train["n_epochs"]
    scaler = GradScaler("cuda")
    total_step = 0
    best_loss = 1e4

    if weighted_loss > 1.0:
        logger.info(f"apply weighted loss = {weighted_loss} on labels: {weighted_loss_label}")

    controlnet.train()
    unet.eval()
    prev_time = time.time()
    for epoch in range(n_epochs):
        epoch_loss_ = 0
        for step, batch in enumerate(train_loader):
            # get embedding and scale if needed
            embeddings = batch["embedding"].to(device) * scale_factor
            labels = batch["segmentation"].to(device) 
            full_h = embeddings.shape[2] * 4
            full_w = embeddings.shape[3] * 4
            full_d = embeddings.shape[4] * 4
            
            labels_resized = F.interpolate(labels.as_tensor().to(torch.uint8), size=(full_h, full_w, full_d), mode="nearest")            #if dim is 4 unsqueeze it
            if labels_resized.dim() == 4:
                labels_resized = labels_resized.unsqueeze(0)
            elif labels_resized.dim() == 6:
                labels_resized = labels_resized.squeeze(0)

            spacing_tensor = batch["spacing"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=True):
                
                controlnet_cond = binarize_labels(labels_resized).float()
                # generate random noise
                noise_shape = list(embeddings.shape)
                noise = torch.randn(noise_shape, dtype=embeddings.dtype).to(device)

                # create timesteps
                if isinstance(noise_scheduler, RFlowScheduler):
                    timesteps = noise_scheduler.sample_timesteps(embeddings)
                else:
                    timesteps = torch.randint(
                        0, noise_scheduler.num_train_timesteps, (embeddings.shape[0],), device=embeddings.device
                    ).long()

                # create noisy latent
                noisy_latent = noise_scheduler.add_noise(original_samples=embeddings, noise=noise, timesteps=timesteps)

                # --- TEXT CONDITIONING ---
                prompts = batch["text_prompt"]
                tokens = tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True).to(device)

                with torch.no_grad():
                    encoder_outputs = text_encoder(**tokens)
                    text_emb = encoder_outputs.last_hidden_state  # [B, seq_len, 768]
                
                controlnet_inputs = {
                    "x": noisy_latent,
                    "timesteps": timesteps,
                    "controlnet_cond": controlnet_cond,
                    "context": text_emb,
                }
                if include_modality:
                    controlnet_inputs.update({"class_labels": torch.ones((len(embeddings),), dtype=torch.long).to(device)})
                                                                                                                  
                down_block_res_samples, mid_block_res_sample = controlnet(**controlnet_inputs)

                # get diffusion network output
                unet_inputs = {
                    "x": noisy_latent,
                    "timesteps": timesteps,
                    "spacing_tensor": spacing_tensor,
                    "down_block_additional_residuals": down_block_res_samples,
                    "mid_block_additional_residual": mid_block_res_sample,
                    "context": text_emb,
                }
                # Add extra arguments if include_body_region is True
                if include_body_region:
                    unet_inputs.update({
                        "top_region_index_tensor": batch["top_region_index"].to(device),
                        "bottom_region_index_tensor": batch["bottom_region_index"].to(device),
                    })
                if include_modality:
                    unet_inputs.update({"class_labels": torch.ones((len(embeddings),), dtype=torch.long).to(device)})
                model_output = unet(**unet_inputs)

            if noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                # predict noise
                model_gt = noise
            elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                # predict sample
                model_gt = embeddings
            elif noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                # predict velocity
                model_gt = embeddings - noise
            else:
                raise ValueError(
                    "noise scheduler prediction type has to be chosen from ",
                    f"[{DDPMPredictionType.EPSILON},{DDPMPredictionType.SAMPLE},{DDPMPredictionType.V_PREDICTION}]",
                )

            if weighted_loss > 1.0:
                weights = torch.ones_like(embeddings).to(embeddings.device)
                roi = torch.zeros([noise_shape[0]] + [1] + noise_shape[2:]).to(embeddings.device)
                # interpolate_label = F.interpolate(batch["label"].to(torch.uint8), size=embeddings.shape[2:], mode="nearest") ## previous code, changed to below
                interpolate_label = F.interpolate(labels.as_tensor().to(torch.uint8), size=embeddings.shape[2:], mode="nearest")            #if dim is 4 unsqueeze it
                # if interpolate_label.dim() == 4:
                #     interpolate_label = interpolate_label.unsqueeze(0)
                # elif interpolate_label.dim() == 6:
                #     interpolate_label = interpolate_label.squeeze(0)

                # assign larger weights for ROI (tumor)
                for label in weighted_loss_label:
                    roi[interpolate_label == label] = 1
                weights[roi.repeat(1, embeddings.shape[1], 1, 1, 1) == 1] = weighted_loss
                # loss = (F.l1_loss(model_output.float(), model_gt.float(), reduction="none") * weights).mean()
                loss = (F.mse_loss(model_output.float(), model_gt.float(), reduction="none") * weights).mean()
            else:
                # loss = F.l1_loss(model_output.float(), model_gt.float())
                loss = F.mse_loss(model_output.float(), model_gt.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            total_step += 1

            if rank == 0:
                # write train loss for each batch into tensorboard
                tensorboard_writer.add_scalar(
                    "train/train_controlnet_loss_iter", loss.detach().cpu().item(), total_step
                )
                batches_done = step + 1
                batches_left = len(train_loader) - batches_done
                time_left = timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()
                logger.info(
                    "\r[Epoch %d/%d] [Batch %d/%d] [LR: %.8f] [loss: %.4f] ETA: %s "
                    % (
                        epoch + 1,
                        n_epochs,
                        step + 1,
                        len(train_loader),
                        lr_scheduler.get_last_lr()[0],
                        loss.detach().cpu().item(),
                        time_left,
                    )
                )
                wandb.log({"epoch": epoch + 1, "batch": step + 1, "loss": loss.detach().cpu().item()})

            epoch_loss_ += loss.detach()

        epoch_loss = epoch_loss_ / (step + 1)

        if use_ddp:
            dist.barrier()
            dist.all_reduce(epoch_loss, op=torch.distributed.ReduceOp.AVG)

        if rank == 0:
            tensorboard_writer.add_scalar("train/train_controlnet_loss_epoch", epoch_loss.cpu().item(), total_step)
            # save controlnet only on master GPU (rank 0)
            controlnet_state_dict = controlnet.module.state_dict() if world_size > 1 else controlnet.state_dict()
            # if args.model_dir doesnt exist, create it
            model_dir = os.path.join(args.model_dir, "weighted_loss_"+str(args.controlnet_train["weighted_loss"]))
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "loss": epoch_loss,
                    "controlnet_state_dict": controlnet_state_dict,
                },
                f"{model_dir}/{args.exp_name}_epoch_{epoch + 1}.pt",
            )

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                logger.info(f"best loss -> {best_loss}.")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "loss": best_loss,
                        "controlnet_state_dict": controlnet_state_dict,
                    },
                    f"{model_dir}/{args.exp_name}_best.pt",
                )

        torch.cuda.empty_cache()
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
