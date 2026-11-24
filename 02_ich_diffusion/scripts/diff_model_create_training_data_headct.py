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

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import monai
import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
from monai.transforms import Compose
from monai.utils import set_determinism

from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .utils import define_instance

### use Lambdad to define own transformation that scale the head ct brain window
from monai.transforms import Lambdad

# Set the random seed for reproducibility
set_determinism(seed=0)

### different from original file: added brain windowing transformation, specific to head ct


# HU breakpoints and their mapped values (same as before)
R_POINTS = torch.tensor([-1000.0, -100.0,   0.0,  80.0, 300.0, 2000.0], dtype=torch.float32,)
S_POINTS = torch.tensor([-1000.0, -600.0, -200.0, 400.0, 700.0, 1000.0], dtype=torch.float32,)

def ct_piecewise_forward_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Piecewise linear transform for head CT in PyTorch.
    Input:  HU tensor x
    Output: warped tensor y (roughly in [-1000, 1000])
    """
    # Make sure breakpoints are on same device/dtype as x
    r = R_POINTS.to(device=x.device, dtype=x.dtype)
    s = S_POINTS.to(device=x.device, dtype=x.dtype)

    y = torch.zeros_like(x)

    # ---- Below first breakpoint: linear extrapolation ----
    below = x <= r[0]
    if below.any():
        a = (s[1] - s[0]) / (r[1] - r[0])  # slope of first segment
        y[below] = s[0] + a * (x[below] - r[0])

    # ---- Above last breakpoint: linear extrapolation ----
    above = x >= r[-1]
    if above.any():
        a = (s[-1] - s[-2]) / (r[-1] - r[-2])  # slope of last segment
        y[above] = s[-1] + a * (x[above] - r[-1])

    # ---- Middle segments ----
    num_segments = r.numel() - 1
    for i in range(num_segments):
        r0, r1 = r[i],   r[i + 1]
        s0, s1 = s[i],   s[i + 1]
        a = (s1 - s0) / (r1 - r0)

        mask = (x > r0) & (x < r1)
        if mask.any():
            y[mask] = s0 + a * (x[mask] - r0)

    # Exact knot points mapped exactly
    for r_k, s_k in zip(r, s):
        eq = x == r_k
        if eq.any():
            y[eq] = s_k

    return y




def create_transforms(dim: tuple = None) -> Compose:
    """
    Create a set of MONAI transforms for preprocessing.

    Args:
        dim (tuple, optional): New dimensions for resizing. Defaults to None.

    Returns:
        Compose: Composed MONAI transforms.
    """
    if dim:
        return Compose(
            [
                monai.transforms.LoadImaged(keys="image"),
                monai.transforms.EnsureChannelFirstd(keys="image"),
                monai.transforms.Orientationd(keys="image", axcodes="RAS"),
                monai.transforms.EnsureTyped(keys="image", dtype=torch.float32),
                monai.transforms.Lambdad(keys="image", func=lambda x: ct_piecewise_forward_torch(x)),
                monai.transforms.ScaleIntensityRanged(
                    keys="image", a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True
                ),
                monai.transforms.Resized(keys="image", spatial_size=dim, mode="trilinear"),
            ]
        )
    else:
        return Compose(
            [
                monai.transforms.LoadImaged(keys="image"),
                monai.transforms.EnsureChannelFirstd(keys="image"),
                monai.transforms.Orientationd(keys="image", axcodes="RAS"),
            ]
        )


def round_number(number: int, base_number: int = 128) -> int:
    """
    Round the number to the nearest multiple of the base number, with a minimum value of the base number.

    Args:
        number (int): Number to be rounded.
        base_number (int): Number to be common divisor.

    Returns:
        int: Rounded number.
    """
    new_number = max(round(float(number) / float(base_number)), 1.0) * float(base_number)
    return int(new_number)


def load_filenames(data_list_path: str) -> list:
    """
    Load filenames from the JSON data list.

    Args:
        data_list_path (str): Path to the JSON data list file.

    Returns:
        list: List of filenames.
    """
    with open(data_list_path, "r") as file:
        json_data = json.load(file)
    filenames_raw = json_data["training"]
    return [_item["image"] for _item in filenames_raw]


def process_file(
    filepath: str,
    args: argparse.Namespace,
    autoencoder: torch.nn.Module,
    device: torch.device,
    plain_transforms: Compose,
    new_transforms: Compose,
    logger: logging.Logger,
) -> None:
    """
    Process a single file to create training data.

    Args:
        filepath (str): Path to the file to be processed.
        args (argparse.Namespace): Configuration arguments.
        autoencoder (torch.nn.Module): Autoencoder model.
        device (torch.device): Device to process the file on.
        plain_transforms (Compose): Plain transforms.
        new_transforms (Compose): New transforms.
        logger (logging.Logger): Logger for logging information.
    """

    out_filename_base = filepath.replace(".gz", "").replace(".nii", "")
    out_filename_base = os.path.join(args.embedding_base_dir, out_filename_base)
    out_filename = out_filename_base + "_emb.nii.gz"

    if os.path.isfile(out_filename):
        return

    test_data = {"image": os.path.join(args.data_base_dir, filepath)}
    transformed_data = plain_transforms(test_data)
    nda = transformed_data["image"]

    dim = [int(nda.meta["dim"][_i]) for _i in range(1, 4)]
    spacing = [float(nda.meta["pixdim"][_i]) for _i in range(1, 4)]

    logger.info(f"old dim: {dim}, old spacing: {spacing}")

    new_data = new_transforms(test_data)
    nda_image = new_data["image"]

    new_affine = nda_image.meta["affine"].numpy()
    nda_image = nda_image.numpy().squeeze()

    logger.info(f"new dim: {nda_image.shape}, new affine: {new_affine}")

    try:
        out_path = Path(out_filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"out_filename: {out_filename}")

        with torch.amp.autocast("cuda"):
            pt_nda = torch.from_numpy(nda_image).float().to(device).unsqueeze(0).unsqueeze(0)
            z = autoencoder.encode_stage_2_inputs(pt_nda)
            logger.info(f"z: {z.size()}, {z.dtype}")

            out_nda = z.squeeze().cpu().detach().numpy().transpose(1, 2, 3, 0)
            out_img = nib.Nifti1Image(np.float32(out_nda), affine=new_affine)
            nib.save(out_img, out_filename)
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")


@torch.inference_mode()
def diff_model_create_training_data(
    env_config_path: str, model_config_path: str, model_def_path: str, num_gpus: int
) -> None:
    """
    Create training data for the diffusion model.

    Args:
        env_config_path (str): Path to the environment configuration file.
        model_config_path (str): Path to the model configuration file.
        model_def_path (str): Path to the model definition file.
    """
    args = load_config(env_config_path, model_config_path, model_def_path)
    local_rank, world_size, device = initialize_distributed(num_gpus=num_gpus)
    logger = setup_logging("creating training data")
    logger.info(f"Using device {device}")

    autoencoder = define_instance(args, "autoencoder_def").to(device)
    try:
        print(args.trained_autoencoder_path)
        checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, weights_only=True)
        autoencoder.load_state_dict(checkpoint_autoencoder)
    except Exception:
        logger.error("The trained_autoencoder_path does not exist!")

    Path(args.embedding_base_dir).mkdir(parents=True, exist_ok=True)
    print("json_data_list:", args.json_data_list)
    filenames_raw = load_filenames(args.json_data_list)
    logger.info(f"filenames_raw: {filenames_raw}")
    logger.info(f"Total number of files to process: {len(filenames_raw)}")

    # print(len(filenames_raw["training"]))

    plain_transforms = create_transforms(dim=None)

    for _iter in range(len(filenames_raw)):
        if _iter % world_size != local_rank:
            continue

        filepath = filenames_raw[_iter]
        new_dim = tuple(
            round_number(
                int(plain_transforms({"image": os.path.join(args.data_base_dir, filepath)})["image"].meta["dim"][_i])
            )
            for _i in range(1, 4)
        )
        # Do this to avoid memory issues for too large images
        if new_dim == (512, 512, 512) or any(d > 512 for d in new_dim):
                new_dim = (512,512,384)
                print(f"new dim: {new_dim}")
        new_transforms = create_transforms(new_dim)

        process_file(filepath, args, autoencoder, device, plain_transforms, new_transforms, logger)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Training Data Creation")
    parser.add_argument(
        "--env_config",
        type=str,
        default="./configs/environment_maisi_diff_model_train.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./configs/config_maisi_diff_model_train.json",
        help="Path to model training/inference configuration",
    )
    parser.add_argument(
        "--model_def", type=str, default="./configs/config_maisi.json", help="Path to model definition file"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for distributed training")

    args = parser.parse_args()
    diff_model_create_training_data(args.env_config, args.model_config, args.model_def, args.num_gpus)
