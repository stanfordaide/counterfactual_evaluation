import json
import os
import subprocess
import multiprocessing as mp
from tqdm import tqdm
import argparse

def download_model_weights(output_dir):
    """Download TotalSegmentator model weights if not present"""
    # Set the model weights directory
    model_dir = os.path.join(output_dir, "model_weights/nnunet/results")
    os.environ['TOTALSEG_HOME_DIR'] = output_dir
    os.makedirs(model_dir, exist_ok=True)
    # Run totalseg_download_weights command
    # cmd = ["totalseg_download_weights", "-t", "total"]
    cmd = ["totalseg_download_weights", "-ta", "lung_vessels"]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Model weights downloaded successfully to {model_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading model weights: {e.stderr}")
        raise

def process_single_case(args):
    """Process a single case with TotalSegmentator"""
    ct_path, output_dir, gpu_id, model_dir = args
    
    # Create output directory for this case
    case_name = os.path.basename(ct_path).split('.')[0]
    case_output_dir = os.path.join(output_dir, f"{case_name}_segmentation_lung_vessels_v2.nii.gz")
    
    # Set the model weights directory for this process
    os.environ['TOTALSEG_HOME_DIR'] = model_dir
    
    # Run TotalSegmentator with specific GPU
    cmd = [
        "TotalSegmentator",
        "-i", ct_path,
        "-o", case_output_dir,
        "--ml",
        "--device", f"gpu:{gpu_id}",
        "--task", "lung_vessels"
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, case_name
    except subprocess.CalledProcessError as e:
        print(f"Error processing {case_name} on GPU {gpu_id}: {e.stderr}")
        return False, case_name

def run_total_segmentator_parallel(dataset_json_path, output_base_dir, num_gpus, ct_dir=None):
    """Run TotalSegmentator in parallel on multiple GPUs"""
    # Set up model weights directory
    model_dir = os.path.join(output_base_dir, "model_weights")
    # print(f"Model directory: {model_dir}")
    # # Download model weights if needed
    # if not os.path.exists(os.path.join(model_dir, "nnunet/results")):
    #     print("Model weights not found. Downloading...")
    #     try:
    #         download_model_weights(model_dir)
    #         print("Model weights downloaded successfully")
    #     except Exception as e:
    #         print("Failed to download model weights. Please check your internet connection and try again.")
    #         return
    
    # Read dataset.json
    with open(dataset_json_path, 'r') as f:
        dataset = json.load(f)

    # Support both {"training": [...]} dict and plain [...] list formats
    if isinstance(dataset, dict):
        items = dataset['training']
    else:
        items = dataset

    # Create output directory
    output_dir = os.path.join(output_base_dir, "total_segmentator_lung_vessels_v2")
    os.makedirs(output_dir, exist_ok=True)

    # Prepare arguments for parallel processing
    if ct_dir is None:
        ct_dir = os.path.join(os.path.dirname(os.path.dirname(dataset_json_path)), "CTPA")
    args_list = []
    for i, item in enumerate(items):
        # Construct path: use 'image_id' if available, otherwise fall back to 'image'
        if 'image_id' in item:
            ct_path = os.path.join(ct_dir, f"{item['image_id']}.nii.gz")
        else:
            ct_path = os.path.join(ct_dir, item['image'])
        if not os.path.exists(ct_path):
            print(f"Warning: File not found: {ct_path}")
            continue
        #if file already in total_segmentator_output_dir:
        case_name = os.path.basename(ct_path).split('.')[0]
        if os.path.exists(os.path.join(output_dir, f"{case_name}_segmentation_lung_vessels_v2.nii.gz")):
            print(f"Skipping {case_name} because it already exists in {output_dir}")
            continue
        # Assign GPU ID in round-robin fashion
        gpu_id = i % num_gpus
        args_list.append((ct_path, output_dir, gpu_id, model_dir))
    
    if not args_list:
        print("No valid files found to process!")
        return
        
    # Set up multiprocessing
    num_processes = min(num_gpus, len(args_list))
    print(f"Running TotalSegmentator on {num_processes} GPUs")
    print(f"Found {len(args_list)} files to process")
    
    # Process cases in parallel
    with mp.Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_case, args_list),
            total=len(args_list),
            desc="Processing cases"
        ))
    
    # Print summary
    successful = sum(1 for success, _ in results if success)
    failed = sum(1 for success, _ in results if not success)
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed cases:")
        for success, case_name in results:
            if not success:
                print(f"- {case_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TotalSegmentator in parallel on multiple GPUs")
    parser.add_argument(
        "--dataset_json",
        type=str,
        # default="/path/to/inspect2/pe_training_data_all/subset_dataset.json",
        default="/path/to/private_data/selected_patients.json",
        help="Path to dataset.json file"
    )
    parser.add_argument(
        "--ct_dir",
        type=str,
        default=None,
        help="Directory containing CT nifti files. If not set, derived from dataset_json path."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/path/to/inspect2/pe_training_data_all",
        help="Base directory for outputs"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=6,
        help="Number of GPUs to use"
    )
    
    args = parser.parse_args()
    
    run_total_segmentator_parallel(
        args.dataset_json,
        args.output_dir,
        args.num_gpus,
        ct_dir=args.ct_dir,
    ) 
