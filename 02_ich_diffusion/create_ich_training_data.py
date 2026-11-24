import os
import sys
from pathlib import Path
import pandas as pd
import json
import subprocess
import argparse
import nibabel as nib
from tqdm import tqdm



def create_dataset_json(all_info_csv_path):

    with open(all_info_csv_path, "r") as f:
        all_info_csv = pd.read_csv(f)
    print(f"Total number of rows in all_info_df_all_non_contrast: {len(all_info_csv)}")

    output_base_dir = os.path.dirname(all_info_csv_path)

    ## The output JSON file needs only the relative paths to the CT files. And it has the format:
    # {
    #     "training": [
    #         {
    #             "image": "Set_08/***/Series_2/Series_2_BRAIN_NONCON_***.nii"
    #         },
    #         {
    #             "image": "Set_00/***/Series_2/Series_2_BRAIN_NONCON_***.nii"
    #         }
    #     ]
    # }

    dataset_list = []
    for _, row in all_info_csv.iterrows():
        image = row['ct_nifti_rel_path']
        dataset_list.append({"image": image})
    dataset_dict = {"training": dataset_list}
    with open(os.path.join(output_base_dir, "ich_training_data_all.json"), "w") as f:
        json.dump(dataset_dict, f, indent=2)
    print(f"Saved training dataset to {os.path.join(output_base_dir, 'ich_training_data_all.json')}")
    print(f"Total number of samples: {len(dataset_list)}")




def extract_vae_features_with_piecewise_linear_mapping(env_config_path: str, model_config_path: str, model_def_path: str, num_gpus: int = 8):

    print("Extracting VAE features using MAISI VAE Model...")
    print("For head ct, we first perform piecewise linear mapping to the ct files, and then extract the VAE features.")
    print(f"Model config: {model_config_path}")
    print(f"Environment config: {env_config_path}")
  
    torchrun_command = [
        "torchrun",
        "--standalone",
        "--nproc_per_node", str(num_gpus),
        "-m", "scripts.diff_model_create_training_data_headct",
        "--env_config", env_config_path,
        "--model_config", model_config_path,
        "--model_def", model_def_path,
        "--num_gpus", str(num_gpus)
    ]
    print("Running:", " ".join(torchrun_command))
    subprocess.run(torchrun_command, check=True)

           


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prepare headct dataset")
    parser.add_argument(
        "--task",
        type=str,
        choices=["create_dataset_json", "extract_vae_features"],
        default="create_dataset_json",
        help="Task to perform",
    )


    parser.add_argument(
        "--env_config",
        type=str,
        default="./configs/environment_maisi_diff_model_headct.json",
        help="Path to environment configuration file",
    )


    parser.add_argument(
        "--model_config",
        type=str,
        default="./configs/config_maisi_diff_model_headct.json",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--model_def",
        type=str,
        default="./configs/config_maisi3d-rflow.json",
        help="Path to model definition file",
    )
    parser.add_argument(
        "--ct_dir",
        type=str,
        default="/path/to/head_ct_niis",
        help="Directory containing CT scans",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    args = parser.parse_args()

    if args.task == "create_dataset_json":
        print("Creating dataset JSON for headct dataset...")
        ## create the training_dataset.json file for the headct dataset. This json file will be used for VAE feature extraction.
        all_info_csv = "/path/to/ich_datalists/all_info.csv"
        create_dataset_json(all_info_csv)



    if args.task == "extract_vae_features":
        
        extract_vae_features_with_piecewise_linear_mapping(
            env_config_path=args.env_config,
            model_config_path=args.model_config,
            model_def_path=args.model_def,
            num_gpus=args.num_gpus,
        ) 
    
