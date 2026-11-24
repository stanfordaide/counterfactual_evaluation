import pandas as pd
from pathlib import Path
import json
import os
import numpy as np
import nibabel as nib
from glob import glob
import argparse
import torch
import subprocess

# Set PyTorch memory management settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

# def setup_distributed(num_gpus):
#     """Set up distributed training environment only if running under a launcher."""
#     if num_gpus > 1 and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
#         os.environ['MASTER_ADDR'] = 'localhost'
#         os.environ['MASTER_PORT'] = '12355'
#         os.environ['WORLD_SIZE'] = str(num_gpus)
#         # Do not set RANK or LOCAL_RANK here, let the launcher do it
#         dist.init_process_group(backend='nccl')
#     else:
#         print("Not running in distributed mode. Proceeding with single process.")

# Data root: $CF_RAW/pe_training_data_all (set the CF_RAW env var).
DATA_ROOT = os.path.join(
    os.environ.get("CF_RAW", "/path/to/private_data"),
    "pe_training_data_all",
)


def create_binary_label(ct_path: str, is_pe_positive: bool, output_dir: str) -> str:
    """
    Create a binary label volume matching the CT dimensions.
    
    Args:
        ct_path: Path to the CT volume
        is_pe_positive: Whether the case is PE positive
        output_dir: Directory to save the label
        
    Returns:
        str: Path to the created label file
    """
    # Load CT to get dimensions
    ct_img = nib.load(ct_path)
    shape = ct_img.shape
    affine = ct_img.affine
    
    # Create binary volume
    label_volume = np.zeros(shape, dtype=np.uint8)
    if is_pe_positive:
        label_volume.fill(1)
        
    # Save as nifti
    out_path = Path(output_dir) / f"{Path(ct_path).stem}_label.nii.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(label_volume, affine), str(out_path))
    
    return str(out_path)

def create_datalist(balanced_df: pd.DataFrame, ct_dir: str, label_dir: str, embedding_dir: str, mode_embedding: str = "balanced") -> dict:
    """
    Create JSON datalist for training using the pre-balanced dataset.
    
    Args:
        balanced_df: DataFrame containing the balanced set of cases
        ct_dir: Directory containing CT scans
        label_dir: Directory containing binary labels
        embedding_dir: Directory containing embeddings
        mode_embedding: Either "balanced" or "all" to determine dataset creation mode
        
    Returns:
        dict: Datalist in required format
    """
    training_list = []
    for _, row in balanced_df.iterrows():
        case_id = row['impression_id']
        ct_path = os.path.join(ct_dir, f"{case_id}.nii.gz")
        embedding_path = os.path.join(embedding_dir, f"{case_id}_emb.nii.gz")
        
        if os.path.exists(ct_path):
            # Use the actual impression text if available, otherwise use default prompt
            text_prompt = row['impressions'] if pd.notna(row['impressions']) else (
                "Evidence of Pulmonary embolism" if int(row['pe_positive']) == 1 else "No pulmonary embolism in this CT"
            )
            
            training_list.append({
                "image": ct_path,
                "label": row['pe_positive'],
                "embedding": embedding_path,
                "pe_status": int(row['pe_positive']),
                "text_prompt": text_prompt
            })
    
    return {"training": training_list}

def create_inspect_dataset_json(ct_dir: str, output_path: str, balanced_df: pd.DataFrame = None) -> None:
    """
    Create a dataset.json file containing paths to selected .nii.gz files.
    
    Args:
        ct_dir: Directory containing CT scans (CTPA folder)
        output_path: Full path where to save the dataset.json file
        balanced_df: DataFrame containing the balanced set of cases to include
    """
    if balanced_df is not None:
        # Use only the balanced set of cases
        dataset_list = []
        for case_id in balanced_df['impression_id']:
            nifti_file = os.path.join(ct_dir, f"{case_id}.nii.gz")
            if os.path.exists(nifti_file):
                # Get relative path from ct_dir
                rel_path = os.path.relpath(nifti_file, ct_dir)
                dataset_list.append({"image": rel_path})
    else:
        # Get all .nii.gz files (fallback to original behavior)
        nifti_files = glob(os.path.join(ct_dir, "*.nii.gz"))
        dataset_list = []
        for nifti_file in nifti_files:
            rel_path = os.path.relpath(nifti_file, ct_dir)
            dataset_list.append({"image": rel_path})
    
    # Create the dataset dictionary
    dataset_dict = {"training": dataset_list}
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(dataset_dict, f, indent=2)
    
    print(f"Created dataset.json at {output_path} with {len(dataset_list)} files")

def prepare_pe_dataset(
    env_config_path: str,
    model_config_path: str,
    model_def_path: str,
    inspect_csv: str,
    impressions_csv: str,
    ct_dir: str,
    output_base_dir: str,
    num_gpus: int = 8,
    debug: bool = False,
    mode_embedding: str = "all"
):
    """
    Prepare the complete PE dataset for ControlNet training.
    
    Args:
        env: Environment to run in ('local' or 'cluster')
        env_config_path: Path to environment config
        model_config_path: Path to model config
        model_def_path: Path to model definition
        inspect_csv: Path to Inspect dataset CSV
        impressions_csv: Path to Final Impressions CSV
        ct_dir: Directory containing CT scans
        output_base_dir: Base directory for outputs
        num_gpus: Number of GPUs to use
        debug: If True, only process 4 cases (2 positive and 2 negative for balanced mode, or 4 random cases for all mode)
        mode_embedding: Either "balanced" or "all" to determine dataset creation mode
    """
    # Set up distributed training if using multiple GPUs
    # setup_distributed(num_gpus)

    # Create output directories
    label_dir = os.path.join(output_base_dir, "labels")
    embedding_dir = os.path.join(output_base_dir, "embeddings")
    print(f"Creating output directories:")
    print(f"  - Base directory: {output_base_dir}")
    print(f"  - Label directory: {label_dir}")
    print(f"  - Embedding directory: {embedding_dir}")
    
    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(embedding_dir, exist_ok=True)
    
    # Read both CSVs
    df = pd.read_csv(inspect_csv)
    impressions_df = pd.read_csv(impressions_csv)
    
    # Merge with impressions
    df = df.merge(impressions_df[['impression_id', 'impressions']], on='impression_id', how='left')
    
    if mode_embedding == "balanced":
        # Split into positive and negative cases
        positive_cases = df[df['pe_positive'] == 1]
        negative_cases = df[df['pe_positive'] == 0]
        
        if debug:
            # For debug mode, take 2 positive and 2 negative cases
            positive_cases = positive_cases.sample(n=2, random_state=42, replace=False)
            negative_cases = negative_cases.sample(n=2, random_state=42, replace=False)
        else:
            # For normal mode, balance the dataset
            n_samples = min(len(positive_cases), len(negative_cases))
            positive_cases = positive_cases.sample(n=n_samples, random_state=42, replace=False)
            negative_cases = negative_cases.sample(n=n_samples, random_state=42, replace=False)
        
        # Combine the balanced dataset
        balanced_df = pd.concat([positive_cases, negative_cases])
    else:  # mode_embedding == "all"
        if debug:
            # For debug mode in all mode, take 3 random cases plus one known case
            known_case = df[df['impression_id'] == 'PE1677861']
            if len(known_case) == 0:
                print("Warning: Known case PE1677861 not found in dataset")
                random_cases = df.sample(n=4, random_state=42, replace=False)
            else:
                random_cases = df[df['impression_id'] != 'PE1677861'].sample(n=3, random_state=42, replace=False)
                balanced_df = pd.concat([known_case, random_cases])
                print(f"Debug mode: Including known case PE1677861 and 3 random cases")
        else:
            balanced_df = df
    
    # Create dataset.json with the selected set of cases
    dataset_json_path = os.path.join(output_base_dir, "dataset.json")
    create_inspect_dataset_json(ct_dir, dataset_json_path, balanced_df)
    print("Dataset.json created with selected set of cases")
    
    # In "all" mode, filter out cases that already have embeddings
    if mode_embedding == "all":
        cases_to_process = []
        for _, row in balanced_df.iterrows():
            case_id = row['impression_id']
            embedding_path = os.path.join(embedding_dir, f"{case_id}_emb.nii.gz")
            if not os.path.exists(embedding_path):
                cases_to_process.append(row)
        
        if cases_to_process:
            print(f"Processing {len(cases_to_process)} new cases out of {len(balanced_df)} total cases")
            # Create temporary dataset.json with only new cases
            temp_df = pd.DataFrame(cases_to_process)
            temp_dataset_json = os.path.join(output_base_dir, "temp_dataset.json")
            create_inspect_dataset_json(ct_dir, temp_dataset_json, temp_df)
            
            # Generate embeddings only for new cases using torchrun
            torchrun_command = [
                "torchrun",
                "--nproc_per_node", str(num_gpus),
                "-m", "scripts.diff_model_create_training_data",
                "--env_config", env_config_path,
                "--model_config", model_config_path,
                "--model_def", model_def_path,
                "--num_gpus", str(num_gpus)
            ]
            print("Running:", " ".join(torchrun_command))
            subprocess.run(torchrun_command, check=True)
            
            # Clean up temporary file
            if os.path.exists(temp_dataset_json):
                os.remove(temp_dataset_json)
        else:
            print("All cases already have embeddings, skipping generation")
    else:
        # In balanced mode, process all cases as before using torchrun
        torchrun_command = [
            "torchrun",
            "--nproc_per_node", str(num_gpus),
            "-m", "scripts.diff_model_create_training_data",
            "--env_config", env_config_path,
            "--model_config", model_config_path,
            "--model_def", model_def_path,
            "--num_gpus", str(num_gpus)
        ]
        print("Running:", " ".join(torchrun_command))
        subprocess.run(torchrun_command, check=True)
    
    # Create datalist using the selected dataset
    datalist = create_datalist(balanced_df, ct_dir, label_dir, embedding_dir, mode_embedding)
    
    # Save datalist
    datalist_path = os.path.join(output_base_dir, "pe_datalist.json")
    with open(datalist_path, 'w') as f:
        json.dump(datalist, f, indent=2)
    
    print(f"Dataset preparation complete. Datalist saved to {datalist_path}")
    print(f"Total samples: {len(datalist['training'])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare PE dataset for ControlNet training")
    parser.add_argument(
        "--env_config",
        type=str,
        default="./configs/environment_maisi_diff_model_text.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./configs/config_maisi_diff_model.json",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--model_def",
        type=str,
        default="./configs/config_maisi3d-rflow.json",
        help="Path to model definition file",
    )
    parser.add_argument(
        "--inspect_csv",
        type=str,
        help="Path to Inspect dataset CSV",
    )
    parser.add_argument(
        "--impressions_csv",
        type=str,
        help="Path to Final Impressions CSV",
    )
    parser.add_argument(
        "--ct_dir",
        type=str,
        help="Directory containing CT scans",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Base directory for outputs",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run in debug mode with only 4 cases",
    )
    parser.add_argument(
        "--mode_embedding",
        type=str,
        choices=["balanced", "all"],
        default="all",
        help="Dataset creation mode: 'balanced' for balanced dataset or 'all' for all cases",
    )
    
    args = parser.parse_args()

    # Default paths derive from the data root (DATA_ROOT; set via the CF_RAW env var).
    if not args.inspect_csv:
        args.inspect_csv = os.path.join(DATA_ROOT, "Final_Impressions_labels.csv")
    if not args.impressions_csv:
        args.impressions_csv = os.path.join(DATA_ROOT, "Final_Impressions.csv")
    if not args.ct_dir:
        args.ct_dir = os.path.join(
            os.environ.get("CF_RAW", "/path/to/private_data"), "CTPA")
    if not args.output_dir:
        args.output_dir = DATA_ROOT
        print(args.output_dir)

    prepare_pe_dataset(
        env_config_path=args.env_config,
        model_config_path=args.model_config,
        model_def_path=args.model_def,
        inspect_csv=args.inspect_csv,
        impressions_csv=args.impressions_csv,
        ct_dir=args.ct_dir,
        output_base_dir=args.output_dir,
        num_gpus=args.num_gpus,
        debug=args.debug,
        mode_embedding=args.mode_embedding
    ) 